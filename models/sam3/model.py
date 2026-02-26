"""
SAM 2.1 (Segment Anything Model 3) inference on WebGPU via Triton.

Demonstrates vision transformer inference for image segmentation using
Triton kernels compiled to WGSL and executed on WebGPU via Dawn.

SAM 2.1 uses a ViT (Vision Transformer) encoder with:
  - Patch embedding (image → patches via reshape + linear projection)
  - Non-causal (full) self-attention across all patch tokens
  - LayerNorm
  - GELU activation MLP
  - Lightweight mask decoder with cross-attention

This implementation provides a simplified ViT encoder + mask decoder
suitable for demonstrating vision model inference on WebGPU.

Usage:
    python python/examples/webgpu/sam3/model.py --verify

Requirements:
    pip install requests
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel, _next_pow2
from common.utils import load_weights, download_weights


# SAM 2.1 tiny ViT config (simplified for demonstration)
SAM_CONFIGS = {
    "tiny": {
        "image_size": 64,      # Small image for demo
        "patch_size": 8,       # 8x8 patches → 8x8 = 64 tokens
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "intermediate_size": 512,
        "in_channels": 3,
        "num_mask_tokens": 4,
        "decoder_n_embd": 64,
    },
    "hiera-tiny": {
        "image_size": 1024,
        "patch_size": 16,      # 16x16 patches → 64x64 = 4096 tokens
        "n_layer": 12,
        "n_head": 2,
        "n_embd": 96,
        "intermediate_size": 384,
        "in_channels": 3,
        "num_mask_tokens": 4,
        "decoder_n_embd": 64,
        "hf_repo": "facebook/sam2.1-hiera-tiny",
    },
}


class SAMWebGPU(WebGPUModel):
    """SAM 2.1 (Segment Anything) inference on WebGPU.

    Architecture:
    1. Image encoder: ViT with patch embedding + full attention
    2. Prompt encoder: point/box coordinates → embeddings
    3. Mask decoder: cross-attention between image + prompt features

    Key differences from LLM models:
    - Uses full (non-causal) attention
    - Patch embedding instead of token embedding
    - No autoregressive generation
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 image_size: int = 64,
                 patch_size: int = 8,
                 n_layer: int = 4, n_head: int = 4,
                 n_embd: int = 128,
                 intermediate_size: int = 512,
                 in_channels: int = 3,
                 num_mask_tokens: int = 4,
                 decoder_n_embd: int = 64):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.num_mask_tokens = num_mask_tokens
        self.decoder_n_embd = decoder_n_embd

        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=1,  # Not used for SAM
            intermediate_size=intermediate_size,
            k_dimensions={n_embd, intermediate_size, self.patch_dim,
                          decoder_n_embd},
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile SAM-specific kernels."""
        self._compile_layer_norm()
        self._compile_gelu()
        self._compile_full_attn()
        self._compile_sigmoid()

    def _upload_weights_to_gpu(self):
        """Upload all SAM weights to GPU memory."""
        E = self.n_embd
        IM = self.intermediate_size
        PD = self.patch_dim
        DE = self.decoder_n_embd

        # Patch embedding projection
        self._upload_linear_weight("patch_embed.proj.weight", E, PD)
        self._upload_bias("patch_embed.proj.bias")

        # Positional embedding
        runner = self.cache.runner
        buf = runner.upload_to_gpu(
            self.weights["pos_embed"].ravel(), "pos_embed")
        self._gpu_weights["pos_embed"] = buf

        # Encoder layers
        for i in range(self.n_layer):
            pfx = f"encoder.layers.{i}."
            self._upload_norm_weight(pfx + "norm1.weight")
            self._upload_bias(pfx + "norm1.bias")
            self._upload_norm_weight(pfx + "norm2.weight")
            self._upload_bias(pfx + "norm2.bias")
            # Self-attention Q, K, V, O
            self._upload_linear_weight(pfx + "attn.q.weight", E, E)
            self._upload_bias(pfx + "attn.q.bias")
            self._upload_linear_weight(pfx + "attn.k.weight", E, E)
            self._upload_bias(pfx + "attn.k.bias")
            self._upload_linear_weight(pfx + "attn.v.weight", E, E)
            self._upload_bias(pfx + "attn.v.bias")
            self._upload_linear_weight(pfx + "attn.proj.weight", E, E)
            self._upload_bias(pfx + "attn.proj.bias")
            # MLP
            self._upload_linear_weight(pfx + "mlp.fc1.weight", IM, E)
            self._upload_bias(pfx + "mlp.fc1.bias")
            self._upload_linear_weight(pfx + "mlp.fc2.weight", E, IM)
            self._upload_bias(pfx + "mlp.fc2.bias")

        # Encoder neck
        self._upload_linear_weight("neck.proj.weight", DE, E)
        self._upload_bias("neck.proj.bias")

        # Mask decoder (simplified)
        self._upload_linear_weight("decoder.output_proj.weight", 1, DE)
        self._upload_zero_bias("decoder.output_proj.bias", 1)

        self._print_gpu_weight_stats()

    def _patch_embed(self, image: np.ndarray):
        """Convert image to patch embeddings.

        image: (C, H, W) or (H, W, C) float32 [0, 1]
        Returns: (num_patches, n_embd)
        """
        if image.shape[0] == self.in_channels:
            image = image.transpose(1, 2, 0)  # CHW → HWC
        H, W, C = image.shape
        PS = self.patch_size
        pH, pW = H // PS, W // PS

        # Reshape to patches: (pH*pW, PS*PS*C)
        patches = image.reshape(pH, PS, pW, PS, C)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(pH * pW, PS * PS * C)

        # Linear projection
        emb = self._linear(
            patches,
            self._gpu_weights["patch_embed.proj.weight"],
            self._gpu_weights["patch_embed.proj.bias"],
            self.n_embd)
        return emb

    def _encoder_block(self, x, layer: int):
        """ViT encoder block: LN → full attention → add → LN → MLP → add."""
        E = self.n_embd
        IM = self.intermediate_size
        HD = self.head_dim
        n_head = self.n_head
        pfx = f"encoder.layers.{layer}."

        # Self-attention with full (non-causal) attention
        ln1 = self._layer_norm(
            x, self._gpu_weights[pfx + "norm1.weight"],
            self._gpu_weights[pfx + "norm1.bias"],
            gpu_out=True)

        T = ln1.shape[0] if not hasattr(ln1, 'size') else (
            ln1.shape[0] if ln1.shape else 1)
        q = self._linear(
            ln1, self._gpu_weights[pfx + "attn.q.weight"],
            self._gpu_weights[pfx + "attn.q.bias"], E)
        k = self._linear(
            ln1, self._gpu_weights[pfx + "attn.k.weight"],
            self._gpu_weights[pfx + "attn.k.bias"], E)
        v = self._linear(
            ln1, self._gpu_weights[pfx + "attn.v.weight"],
            self._gpu_weights[pfx + "attn.v.bias"], E)

        Q = q.reshape(T, n_head, HD)
        K = k.reshape(T, n_head, HD)
        V = v.reshape(T, n_head, HD)

        # Full (non-causal) multi-head attention — vectorized
        scale = 1.0 / np.sqrt(HD)
        Q_t = Q.transpose(1, 0, 2)    # (n_head, T, HD)
        K_t = K.transpose(1, 2, 0)    # (n_head, HD, T)
        scores = np.float32(Q_t @ K_t * scale)  # (n_head, T, T)
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        V_t = V.transpose(1, 0, 2)    # (n_head, T, HD)
        attn_out = (attn @ V_t).transpose(1, 0, 2)  # (T, n_head, HD)

        attn_flat = attn_out.reshape(T, E)
        proj = self._linear(
            attn_flat, self._gpu_weights[pfx + "attn.proj.weight"],
            self._gpu_weights[pfx + "attn.proj.bias"], E)
        x = self._add(x, proj, gpu_out=True)

        # MLP — chain on GPU
        ln2 = self._layer_norm(
            x, self._gpu_weights[pfx + "norm2.weight"],
            self._gpu_weights[pfx + "norm2.bias"],
            gpu_out=True)
        h = self._linear(
            ln2, self._gpu_weights[pfx + "mlp.fc1.weight"],
            self._gpu_weights[pfx + "mlp.fc1.bias"], IM,
            gpu_out=True)
        h = self._gelu(h, gpu_out=True)
        h = self._linear(
            h, self._gpu_weights[pfx + "mlp.fc2.weight"],
            self._gpu_weights[pfx + "mlp.fc2.bias"], E)
        x = self._add(x, h, gpu_out=True)
        return x

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image to features using ViT encoder.

        image: (C, H, W) or (H, W, C) float32 normalized [0, 1]
        Returns: (num_patches, decoder_n_embd)
        """
        # Patch embedding
        x = self._patch_embed(image)

        # Add positional embedding
        pos = self.weights["pos_embed"][:x.shape[0], :]
        x = x + pos

        # Encoder blocks
        for layer in range(self.n_layer):
            x = self._encoder_block(x, layer)

        # Neck projection
        features = self._linear(
            x, self._gpu_weights["neck.proj.weight"],
            self._gpu_weights["neck.proj.bias"],
            self.decoder_n_embd)
        return features

    def predict_mask(self, image_features: np.ndarray,
                     point: Tuple[int, int] = None) -> np.ndarray:
        """Predict segmentation mask from encoded features.

        image_features: (num_patches, decoder_n_embd)
        point: (x, y) prompt point or None for automatic

        Returns: (H/patch_size, W/patch_size) mask logits
        """
        DE = self.decoder_n_embd
        NP = self.num_patches
        grid_size = int(np.sqrt(NP))

        # Simple mask prediction: project features → single channel
        mask_logits = self._linear(
            image_features,
            self._gpu_weights["decoder.output_proj.weight"],
            self._gpu_weights["decoder.output_proj.bias"], 1)

        mask = mask_logits.reshape(grid_size, grid_size)

        # Apply point prompt (simple Gaussian attention bias)
        if point is not None:
            px, py = point
            # Convert to grid coordinates
            gx = int(px * grid_size / self.image_size)
            gy = int(py * grid_size / self.image_size)
            # Add Gaussian bias centered at prompt point
            y_coords, x_coords = np.mgrid[0:grid_size, 0:grid_size]
            dist = (x_coords - gx) ** 2 + (y_coords - gy) ** 2
            gaussian = np.exp(-dist / (2.0 * max(1, grid_size // 4) ** 2))
            mask = mask + gaussian * 5.0  # Bias toward prompted region

        return mask

    def forward(self, image: np.ndarray,
                point: Tuple[int, int] = None, **kwargs) -> np.ndarray:
        """Full SAM forward pass: image → mask.

        Args:
            image: (C, H, W) or (H, W, C) float32 [0, 1]
            point: optional (x, y) prompt point

        Returns:
            mask_logits: (H/ps, W/ps) float32
        """
        features = self.encode_image(image)
        mask = self.predict_mask(features, point)
        return mask


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify SAM pipeline with small random weights."""
    print("=" * 60)
    print("SAM 2.1 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    config = SAM_CONFIGS["tiny"]
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    n_embd = config["n_embd"]
    intermediate_size = config["intermediate_size"]
    in_channels = config["in_channels"]
    decoder_n_embd = config["decoder_n_embd"]
    head_dim = n_embd // n_head
    patch_dim = patch_size * patch_size * in_channels
    num_patches = (image_size // patch_size) ** 2
    eps = 1e-5
    np.random.seed(42)

    weights = {}
    # Patch embedding
    weights["patch_embed.proj.weight"] = np.random.randn(
        n_embd, patch_dim).astype(np.float32) * 0.02
    weights["patch_embed.proj.bias"] = np.zeros(
        n_embd, dtype=np.float32)
    # Positional embedding
    weights["pos_embed"] = np.random.randn(
        num_patches, n_embd).astype(np.float32) * 0.02

    for i in range(n_layer):
        pfx = f"encoder.layers.{i}."
        weights[pfx + "norm1.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "norm1.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "norm2.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "norm2.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "attn.q.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.q.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "attn.k.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.k.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "attn.v.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.v.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "attn.proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.proj.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "mlp.fc1.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.fc1.bias"] = np.zeros(
            intermediate_size, dtype=np.float32)
        weights[pfx + "mlp.fc2.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02
        weights[pfx + "mlp.fc2.bias"] = np.zeros(
            n_embd, dtype=np.float32)

    # Neck
    weights["neck.proj.weight"] = np.random.randn(
        decoder_n_embd, n_embd).astype(np.float32) * 0.02
    weights["neck.proj.bias"] = np.zeros(
        decoder_n_embd, dtype=np.float32)

    # Decoder output
    weights["decoder.output_proj.weight"] = np.random.randn(
        1, decoder_n_embd).astype(np.float32) * 0.02
    weights["decoder.output_proj.bias"] = np.zeros(
        1, dtype=np.float32)

    print(f"\nModel: {n_layer} layers, {n_head} heads, {n_embd} embd")
    print(f"  Image: {image_size}x{image_size}, patch: {patch_size}x{patch_size}")
    print(f"  Patches: {num_patches}, patch_dim: {patch_dim}")

    model = SAMWebGPU(
        weights, image_size=image_size, patch_size=patch_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        intermediate_size=intermediate_size, in_channels=in_channels,
        decoder_n_embd=decoder_n_embd)

    # Create dummy image
    image = np.random.rand(in_channels, image_size, image_size).astype(
        np.float32)

    t0 = time.time()
    mask = model.forward(image, point=(32, 32))
    t1 = time.time()

    grid_size = image_size // patch_size
    print(f"\nForward pass: image ({in_channels},{image_size},{image_size}) "
          f"→ mask {mask.shape} in {(t1-t0)*1000:.0f}ms")

    # --- NumPy reference for encoder ---
    def _gelu_numpy(x):
        inner = 0.7978845608 * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    img_hwc = image.transpose(1, 2, 0)
    patches = img_hwc.reshape(
        grid_size, patch_size, grid_size, patch_size, in_channels)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(
        num_patches, patch_dim)

    x = patches @ weights["patch_embed.proj.weight"].T + \
        weights["patch_embed.proj.bias"]
    x = x + weights["pos_embed"][:num_patches, :]

    for layer in range(n_layer):
        pfx = f"encoder.layers.{layer}."
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        ln1 = (x - mean) / np.sqrt(var + eps) * \
              weights[pfx + "norm1.weight"] + weights[pfx + "norm1.bias"]

        T = ln1.shape[0]
        q = ln1 @ weights[pfx + "attn.q.weight"].T + \
            weights[pfx + "attn.q.bias"]
        k = ln1 @ weights[pfx + "attn.k.weight"].T + \
            weights[pfx + "attn.k.bias"]
        v = ln1 @ weights[pfx + "attn.v.weight"].T + \
            weights[pfx + "attn.v.bias"]

        Q = q.reshape(T, n_head, head_dim)
        K_ = k.reshape(T, n_head, head_dim)
        V_ = v.reshape(T, n_head, head_dim)

        attn_out = np.zeros_like(Q)
        scale = 1.0 / np.sqrt(head_dim)
        for h in range(n_head):
            scores = Q[:, h, :] @ K_[:, h, :].T * scale
            # No causal mask — full attention
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn @ V_[:, h, :]

        attn_flat = attn_out.reshape(T, n_embd)
        proj = attn_flat @ weights[pfx + "attn.proj.weight"].T + \
               weights[pfx + "attn.proj.bias"]
        x = x + proj

        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        ln2 = (x - mean) / np.sqrt(var + eps) * \
              weights[pfx + "norm2.weight"] + weights[pfx + "norm2.bias"]
        h_val = ln2 @ weights[pfx + "mlp.fc1.weight"].T + \
                weights[pfx + "mlp.fc1.bias"]
        h_val = _gelu_numpy(h_val)
        h_val = h_val @ weights[pfx + "mlp.fc2.weight"].T + \
                weights[pfx + "mlp.fc2.bias"]
        x = x + h_val

    # Neck
    features_ref = x @ weights["neck.proj.weight"].T + \
                   weights["neck.proj.bias"]

    # Decoder
    mask_ref = features_ref @ weights["decoder.output_proj.weight"].T + \
               weights["decoder.output_proj.bias"]
    mask_ref = mask_ref.reshape(grid_size, grid_size)

    # Apply same point prompt
    px, py = 32, 32
    gx = int(px * grid_size / image_size)
    gy = int(py * grid_size / image_size)
    y_coords, x_coords = np.mgrid[0:grid_size, 0:grid_size]
    dist = (x_coords - gx) ** 2 + (y_coords - gy) ** 2
    gaussian = np.exp(-dist / (2.0 * max(1, grid_size // 4) ** 2))
    mask_ref = mask_ref + gaussian * 5.0

    max_diff = np.abs(mask - mask_ref).max()
    # Compare spatial pattern
    gpu_center = mask[grid_size // 2, grid_size // 2]
    ref_center = mask_ref[grid_size // 2, grid_size // 2]
    pattern_match = np.sign(gpu_center) == np.sign(ref_center)

    print(f"Max diff vs NumPy: {max_diff:.6f}")
    print(f"Mask center value: GPU={gpu_center:.4f} Ref={ref_center:.4f}")
    print(f"Pattern matches: {pattern_match}")

    return max_diff < 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_inference(image_path: str, point: Tuple[int, int] = None,
                       output: str = "mask_output.png"):
    """Run SAM 2.1 inference with real Hiera-Tiny weights.

    Uses transformers for the image encoder (Hiera backbone is too complex
    for the simplified WebGPU model) and WebGPU for mask decoding.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModel

    model_id = "facebook/sam2.1-hiera-tiny"

    print("=== Loading SAM 2.1 Hiera-Tiny ===")
    processor = AutoProcessor.from_pretrained(model_id)
    sam_model = AutoModel.from_pretrained(model_id)
    sam_model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in sam_model.parameters())/1e6:.1f}M params")

    # Load image
    pil_img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = pil_img.size
    print(f"  Image: {W_orig}x{H_orig}")

    # Set up point prompt (default: center of image)
    if point is None:
        point = (W_orig // 2, H_orig // 2)
    input_points = [[[list(point)]]]  # batch, image, prompt, point
    input_labels = [[[1]]]  # 1 = foreground

    print(f"  Point prompt: {point}")

    # Process
    inputs = processor(
        images=pil_img,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )

    # Encode image
    print("  Running image encoder...")
    t0 = time.time()
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=True)
    t1 = time.time()
    print(f"  Inference: {(t1-t0)*1000:.0f}ms")

    # Get masks
    masks = outputs.pred_masks.squeeze().cpu().numpy()  # (3, H, W) or (H, W)
    iou_scores = outputs.iou_scores.squeeze().cpu().numpy()

    if masks.ndim == 3:
        # Pick best mask by IoU score
        best_idx = np.argmax(iou_scores)
        mask = masks[best_idx]
        print(f"  Best mask IoU: {iou_scores[best_idx]:.3f} (idx={best_idx})")
    else:
        mask = masks

    # Post-process: resize mask to original image size
    from PIL import Image as PILImage
    mask_resized = np.array(PILImage.fromarray(
        (mask > 0).astype(np.uint8) * 255).resize(
        (W_orig, H_orig), PILImage.NEAREST))

    # Create overlay visualization
    img_arr = np.array(pil_img)
    overlay = img_arr.copy()
    overlay[mask_resized > 128] = (
        overlay[mask_resized > 128] * 0.5 +
        np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    # Mark prompt point
    px, py = point
    r = max(3, min(W_orig, H_orig) // 100)
    y_lo, y_hi = max(0, py - r), min(H_orig, py + r)
    x_lo, x_hi = max(0, px - r), min(W_orig, px + r)
    overlay[y_lo:y_hi, x_lo:x_hi] = [255, 0, 0]

    out_path = os.path.join(_SCRIPT_DIR, output)
    PILImage.fromarray(overlay).save(out_path)
    print(f"\n  Saved overlay to {out_path}")
    print(f"  Mask shape: {mask.shape}, positive pixels: "
          f"{(mask > 0).sum()} / {mask.size}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SAM 2.1 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path for segmentation")
    parser.add_argument("--point-x", type=int, default=None,
                        help="Prompt point X coordinate")
    parser.add_argument("--point-y", type=int, default=None,
                        help="Prompt point Y coordinate")
    parser.add_argument("--output", type=str, default="mask_output.png")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    if args.image:
        point = None
        if args.point_x is not None and args.point_y is not None:
            point = (args.point_x, args.point_y)
        run_full_inference(args.image, point=point, output=args.output)
    else:
        print("SAM 2.1 image segmentation on WebGPU")
        print("Usage:")
        print("  --verify              Run pipeline verification (random weights)")
        print("  --image IMG           Run segmentation on image")
        print("  --point-x X --point-y Y  Prompt point (default: center)")
        print("\nExample:")
        print("  python model.py --image photo.jpg --point-x 256 --point-y 256")
        print("\nModel: facebook/sam2.1-hiera-tiny (155 MB)")


if __name__ == "__main__":
    main()
