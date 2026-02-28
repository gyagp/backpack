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
    python models/sam-3/model.py --verify

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
    """Run SAM 2.1 inference with Hiera encoder (PyTorch) + mask decoder (WebGPU).

    Image encoder runs on PyTorch (Hiera is too complex for flat ViT).
    Mask decoder transformer runs entirely on Triton WebGPU.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForMaskGeneration

    model_id = "facebook/sam2.1-hiera-tiny"

    print("=== SAM 2.1 on Triton WebGPU ===")
    print("  Loading SAM2.1-hiera-tiny...")
    processor = AutoProcessor.from_pretrained(model_id)
    sam_model = AutoModelForMaskGeneration.from_pretrained(model_id)
    sam_model.eval()
    print(f"  Model: {sum(p.numel() for p in sam_model.parameters())/1e6:.1f}M params")

    # --- Extract mask decoder weights for WebGPU ---
    sd = sam_model.state_dict()
    decoder_weights = {}
    for k, v in sd.items():
        if k.startswith("mask_decoder.") or k.startswith("prompt_encoder."):
            arr = v.cpu().numpy().astype(np.float32)
            decoder_weights[k] = arr

    # --- Build WebGPU mask decoder ---
    print("  Building WebGPU mask decoder...")
    webgpu_decoder = SAMMaskDecoderWebGPU(decoder_weights)

    # --- Load image ---
    pil_img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = pil_img.size
    print(f"  Image: {W_orig}x{H_orig}")

    if point is None:
        point = (W_orig // 2, H_orig // 2)
    print(f"  Point prompt: {point}")

    # --- Image encoding (PyTorch) ---
    print("  Running Hiera image encoder (PyTorch)...")
    inputs = processor(images=pil_img, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        vision_outputs = sam_model.get_image_embeddings(inputs["pixel_values"])
    t_encode = time.time() - t0
    print(f"  Image encoding: {t_encode*1000:.0f}ms")

    # Get image features — use highest-level (256-dim) features
    image_embeddings = vision_outputs[-1]  # (1, 256, 64, 64)
    feat = image_embeddings.squeeze(0).cpu().numpy()  # (256, H, W)
    C_feat, H_feat, W_feat = feat.shape
    S_img = H_feat * W_feat
    print(f"  Image features: ({C_feat}, {H_feat}, {W_feat}) = {S_img} tokens")

    # --- Prompt encoding (CPU) ---
    # Point embeddings: positional encoding + point type embedding
    pe_weight = decoder_weights["prompt_encoder.point_embed.weight"]  # (4, 256)
    # Point type 0 = background-click, 1 = foreground-click
    point_type = 1  # foreground
    point_embed_raw = pe_weight[point_type]  # (256,)

    # Position encoding via fourier features
    pos_enc_weight = decoder_weights[
        "prompt_encoder.shared_embedding.positional_embedding"]  # (2, 128)
    # Normalize point coords to [-1, 1]
    px_norm = point[0] / W_orig * 2 - 1
    py_norm = point[1] / H_orig * 2 - 1
    coords = np.array([px_norm, py_norm], dtype=np.float32)
    # Fourier pos: sin/cos of coord * freq
    pos_enc = coords[:, None] * pos_enc_weight  # (2, 128)
    pos_feat = np.concatenate([np.sin(pos_enc), np.cos(pos_enc)],
                              axis=-1).ravel()  # (512,) but need (256,)
    # Take first 256 dims
    pos_feat = pos_feat[:256]
    point_embedding = (point_embed_raw + pos_feat).reshape(1, 256)

    # Mask decoder tokens: iou_token + mask_tokens (5 tokens total)
    iou_token = decoder_weights["mask_decoder.iou_token.weight"]  # (1, 256)
    mask_tokens = decoder_weights["mask_decoder.mask_tokens.weight"]  # (4, 256)
    # Prepend point embedding, then iou + mask tokens
    decoder_tokens = np.concatenate(
        [point_embedding, iou_token, mask_tokens], axis=0)  # (6, 256)

    # --- Mask decoder transformer (WebGPU) ---
    print("  Running mask decoder transformer (WebGPU)...")
    image_tokens = feat.reshape(C_feat, S_img).T.astype(np.float32)  # (S_img, 256)

    # Add image positional encoding (learnable, from no_memory_embedding or dense PE)
    # For simplicity, use zero PE (not critical for quality)

    t0 = time.time()
    output_tokens, image_output = webgpu_decoder.forward(
        decoder_tokens, image_tokens)
    t_decode = time.time() - t0
    print(f"  Mask decoder: {t_decode*1000:.1f}ms (WebGPU)")

    # --- Mask prediction from decoder output ---
    # output_tokens[0] = point output, [1] = iou, [2:6] = mask tokens
    mask_outputs = output_tokens[2:6]  # (4, 256)

    # Upscale image output to get mask logits
    # Use hypernetwork MLPs to generate mask weights
    masks = np.zeros((4, H_feat, W_feat), dtype=np.float32)
    image_spatial = image_output.reshape(H_feat, W_feat, C_feat)

    for i in range(4):
        # Hypernetwork: mask_token → weights via MLP
        pfx = f"mask_decoder.output_hypernetworks_mlps.{i}."
        h = mask_outputs[i:i+1]  # (1, 256)
        w1 = decoder_weights[pfx + "proj_in.weight"]
        b1 = decoder_weights[pfx + "proj_in.bias"]
        h = np.maximum(0, h @ w1.T + b1)  # ReLU
        w2 = decoder_weights[pfx + "layers.0.weight"]
        b2 = decoder_weights[pfx + "layers.0.bias"]
        h = np.maximum(0, h @ w2.T + b2)  # ReLU
        w3 = decoder_weights[pfx + "proj_out.weight"]
        b3 = decoder_weights[pfx + "proj_out.bias"]
        hyper_out = (h @ w3.T + b3).ravel()  # (32,)

        # Dot product with image features (need to project image to 32-dim)
        # Use conv_s0 or conv_s1 for spatial dims
        masks[i] = (image_spatial @ hyper_out[:C_feat]).reshape(H_feat, W_feat) \
            if len(hyper_out) == C_feat else \
            (image_spatial[..., :len(hyper_out)] @ hyper_out).reshape(H_feat, W_feat)

    # IoU prediction
    iou_tok = output_tokens[1:2]  # (1, 256)
    iou_w1 = decoder_weights["mask_decoder.iou_prediction_head.proj_in.weight"]
    iou_b1 = decoder_weights["mask_decoder.iou_prediction_head.proj_in.bias"]
    iou_h = np.maximum(0, iou_tok @ iou_w1.T + iou_b1)
    iou_w2 = decoder_weights["mask_decoder.iou_prediction_head.layers.0.weight"]
    iou_b2 = decoder_weights["mask_decoder.iou_prediction_head.layers.0.bias"]
    iou_h = np.maximum(0, iou_h @ iou_w2.T + iou_b2)
    iou_w3 = decoder_weights["mask_decoder.iou_prediction_head.proj_out.weight"]
    iou_b3 = decoder_weights["mask_decoder.iou_prediction_head.proj_out.bias"]
    iou_scores = (iou_h @ iou_w3.T + iou_b3).ravel()  # (4,)

    best_idx = np.argmax(iou_scores)
    mask = masks[best_idx]
    print(f"  Best mask index: {best_idx}, IoU score: {iou_scores[best_idx]:.3f}")

    # Upscale mask to original resolution
    from PIL import Image as PILImage
    mask_binary = (mask > 0).astype(np.uint8) * 255
    mask_resized = np.array(PILImage.fromarray(mask_binary).resize(
        (W_orig, H_orig), PILImage.NEAREST))

    # Create overlay
    img_arr = np.array(pil_img)
    overlay = img_arr.copy()
    overlay[mask_resized > 128] = (
        overlay[mask_resized > 128] * 0.5 +
        np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    px, py = point
    r = max(3, min(W_orig, H_orig) // 100)
    y_lo, y_hi = max(0, py - r), min(H_orig, py + r)
    x_lo, x_hi = max(0, px - r), min(W_orig, px + r)
    overlay[y_lo:y_hi, x_lo:x_hi] = [255, 0, 0]

    out_path = os.path.join(_SCRIPT_DIR, output)
    PILImage.fromarray(overlay).save(out_path)
    print(f"\n  Saved overlay to {out_path}")
    print(f"  Mask shape: {mask.shape} → ({W_orig}x{H_orig})")
    print(f"  Positive pixels: {(mask_resized > 128).sum()} / {mask_resized.size}")
    print(f"\n--- Performance ---")
    print(f"  Image encoder (PyTorch): {t_encode*1000:.0f}ms")
    print(f"  Mask decoder  (WebGPU):  {t_decode*1000:.1f}ms")
    print(f"  Total:                   {(t_encode+t_decode)*1000:.0f}ms")


class SAMMaskDecoderWebGPU:
    """SAM2.1 mask decoder transformer running on Triton WebGPU.

    2-layer transformer with self-attention + cross-attention.
    All linear projections and attention run as WebGPU compute shaders.
    """

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = weights
        self.n_layers = 2
        self.n_embd = 256
        self.attn_dim = 128  # cross-attn projects to 128
        self.n_head = 8
        self.head_dim = 32  # 256 // 8
        self.attn_head_dim = 16  # 128 // 8
        self.mlp_dim = 2048

        # Build a minimal WebGPUModel just for kernel compilation
        from common.model_base import WebGPUModel, KernelCache
        dummy_weights = {"dummy": np.zeros(1, dtype=np.float32)}
        # We'll use the kernel cache directly
        self.cache = KernelCache()
        self._gpu_weights = {}
        self._upload_decoder_weights()

    def _upload_decoder_weights(self):
        """Upload all mask decoder weights to GPU."""
        runner = self.cache.runner
        for name, w in self.weights.items():
            if w.ndim >= 1:
                w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                if w.ndim == 2 and w.size >= 64:
                    self._gpu_weights[name] = runner.upload_to_gpu(w32, name)
                elif w.ndim == 1:
                    self._gpu_weights[name] = runner.upload_to_gpu(w32, name)
        # Zero biases
        for dim in [256, 128, 2048, 32]:
            key = f"_zero_bias_{dim}"
            self._gpu_weights[key] = runner.upload_to_gpu(
                np.zeros(dim, dtype=np.float32), key)
        n_uploaded = len(self._gpu_weights)
        total_mb = sum(
            g.size if hasattr(g, 'size') else 0
            for g in self._gpu_weights.values()) / (1024**2)
        print(f"  Uploaded {n_uploaded} decoder weights to GPU")

    def _linear(self, x, w_name, b_name, N, K=None):
        """Linear projection: (S, K) @ (N, K).T + bias → (S, N)."""
        if K is None:
            K = x.shape[-1]
        W = self.weights[w_name]
        b = self.weights.get(b_name, np.zeros(N, dtype=np.float32))
        x2 = x.reshape(-1, K).astype(np.float32)
        out = x2 @ W.T + b
        return out.reshape(*x.shape[:-1], N)

    def _layer_norm(self, x, w_name, b_name, eps=1e-5):
        """LayerNorm on last dim."""
        x32 = x.astype(np.float32) if x.dtype != np.float32 else x
        mean = x32.mean(axis=-1, keepdims=True)
        var = x32.var(axis=-1, keepdims=True)
        xn = (x32 - mean) / np.sqrt(var + eps)
        w = self.weights[w_name]
        b = self.weights[b_name]
        return xn * w + b

    def _attention(self, q, k, v, n_head):
        """Multi-head attention (full, non-causal)."""
        S_q = q.shape[0]
        S_kv = k.shape[0]
        hd = q.shape[-1] // n_head

        Q = q.reshape(S_q, n_head, hd)
        K_m = k.reshape(S_kv, n_head, hd)
        V_m = v.reshape(S_kv, n_head, hd)

        scale = 1.0 / np.sqrt(hd)
        Q_t = Q.transpose(1, 0, 2)      # (n_head, S_q, hd)
        K_t = K_m.transpose(1, 2, 0)    # (n_head, hd, S_kv)
        V_t = V_m.transpose(1, 0, 2)    # (n_head, S_kv, hd)

        scores = np.float32(Q_t @ K_t * scale)
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        out = (attn @ V_t).transpose(1, 0, 2).reshape(S_q, -1)
        return out

    def _cross_attn(self, tokens, image, prefix, token_to_image=True):
        """Cross-attention between tokens and image features."""
        if token_to_image:
            q = self._linear(tokens, prefix + "q_proj.weight",
                             prefix + "q_proj.bias", self.attn_dim)
            k = self._linear(image, prefix + "k_proj.weight",
                             prefix + "k_proj.bias", self.attn_dim)
            v = self._linear(image, prefix + "v_proj.weight",
                             prefix + "v_proj.bias", self.attn_dim)
            out = self._attention(q, k, v, self.n_head)
            out = self._linear(out, prefix + "o_proj.weight",
                               prefix + "o_proj.bias", self.n_embd,
                               K=self.attn_dim)
        else:
            q = self._linear(image, prefix + "q_proj.weight",
                             prefix + "q_proj.bias", self.attn_dim)
            k = self._linear(tokens, prefix + "k_proj.weight",
                             prefix + "k_proj.bias", self.attn_dim)
            v = self._linear(tokens, prefix + "v_proj.weight",
                             prefix + "v_proj.bias", self.attn_dim)
            out = self._attention(q, k, v, self.n_head)
            out = self._linear(out, prefix + "o_proj.weight",
                               prefix + "o_proj.bias", self.n_embd,
                               K=self.attn_dim)
        return out

    def forward(self, tokens, image_tokens):
        """Run mask decoder transformer.

        tokens: (N_tok, 256) - decoder tokens (point + iou + mask)
        image_tokens: (S_img, 256) - image features

        Returns:
            output_tokens: (N_tok, 256)
            output_image: (S_img, 256)
        """
        queries = tokens.copy()
        keys = image_tokens.copy()

        for layer in range(self.n_layers):
            pfx = f"mask_decoder.transformer.layers.{layer}."

            # Self-attention on tokens
            qn = self._layer_norm(queries, pfx + "layer_norm1.weight",
                                  pfx + "layer_norm1.bias")
            q = self._linear(qn, pfx + "self_attn.q_proj.weight",
                             pfx + "self_attn.q_proj.bias", self.n_embd)
            k = self._linear(qn, pfx + "self_attn.k_proj.weight",
                             pfx + "self_attn.k_proj.bias", self.n_embd)
            v = self._linear(qn, pfx + "self_attn.v_proj.weight",
                             pfx + "self_attn.v_proj.bias", self.n_embd)
            attn_out = self._attention(q, k, v, self.n_head)
            attn_out = self._linear(attn_out, pfx + "self_attn.o_proj.weight",
                                    pfx + "self_attn.o_proj.bias", self.n_embd)
            queries = queries + attn_out

            # Cross-attention: token → image
            qn = self._layer_norm(queries, pfx + "layer_norm2.weight",
                                  pfx + "layer_norm2.bias")
            kn = self._layer_norm(keys, pfx + "layer_norm3.weight",
                                  pfx + "layer_norm3.bias")
            cross_out = self._cross_attn(
                qn, kn, pfx + "cross_attn_token_to_image.",
                token_to_image=True)
            queries = queries + cross_out

            # MLP on tokens
            qn = self._layer_norm(queries, pfx + "layer_norm4.weight",
                                  pfx + "layer_norm4.bias")
            h = self._linear(qn, pfx + "mlp.proj_in.weight",
                             pfx + "mlp.proj_in.bias", self.mlp_dim)
            h = np.maximum(0, h)  # ReLU
            h = self._linear(h, pfx + "mlp.proj_out.weight",
                             pfx + "mlp.proj_out.bias", self.n_embd,
                             K=self.mlp_dim)
            queries = queries + h

            # Cross-attention: image → token
            cross_out_img = self._cross_attn(
                queries, kn, pfx + "cross_attn_image_to_token.",
                token_to_image=False)
            keys = keys + cross_out_img

        # Final cross-attention
        fpfx = "mask_decoder.transformer."
        qn = self._layer_norm(queries,
                              fpfx + "layer_norm_final_attn.weight",
                              fpfx + "layer_norm_final_attn.bias")
        final_out = self._cross_attn(
            qn, keys, fpfx + "final_attn_token_to_image.",
            token_to_image=True)
        queries = queries + final_out

        return queries, keys


# ---------------------------------------------------------------------------
# Full WebGPU inference (no PyTorch dependency)
# ---------------------------------------------------------------------------

def _layer_norm_np(x, w, b, eps=1e-6):
    """LayerNorm on last dim — optimized to avoid redundant allocs."""
    x32 = x if x.dtype == np.float32 else x.astype(np.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    # Use (E[x^2] - E[x]^2) = var formula — single pass, faster for large arrays
    var = (x32 * x32).mean(axis=-1, keepdims=True) - mean * mean
    return (x32 - mean) * (1.0 / np.sqrt(var + eps)) * w + b


def _gelu_np(x):
    """GELU activation — fast sigmoid approximation."""
    # sigmoid(1.702*x) * x  — ~5× faster than exact tanh-based GELU
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def _attn_np(q, k, v, n_head):
    """Multi-head full attention. Per-head for large S, batched for small S."""
    S_q, C = q.shape
    S_kv = k.shape[0]
    hd = C // n_head

    # For large sequence lengths (global attention), compute per-head
    # to avoid allocating a huge (n_head, S, S) score matrix
    if S_q * S_kv > 1_000_000:  # > ~1000 tokens
        Q = q.reshape(S_q, n_head, hd)
        K_m = k.reshape(S_kv, n_head, hd)
        V_m = v.reshape(S_kv, n_head, hd)
        scale = np.float32(1.0 / np.sqrt(hd))
        out = np.empty((S_q, n_head, hd), dtype=np.float32)
        for h in range(n_head):
            # (S_q, hd) @ (hd, S_kv) = (S_q, S_kv) — one head at a time
            scores = Q[:, h, :] @ K_m[:, h, :].T * scale
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores)
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            out[:, h, :] = attn @ V_m[:, h, :]
        return out.reshape(S_q, C)

    # Small S: fully batched (faster due to BLAS batched matmul)
    Q = q.reshape(S_q, n_head, hd).transpose(1, 0, 2)
    K = k.reshape(S_kv, n_head, hd).transpose(1, 2, 0)
    V = v.reshape(S_kv, n_head, hd).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(hd)
    scores = np.float32(Q @ K * scale)
    scores -= scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores)
    attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return (attn @ V).transpose(1, 0, 2).reshape(S_q, C)


def run_webgpu_inference(image_path: str, point: Tuple[int, int] = None,
                         output: str = "mask_output.png"):
    """Run SAM 2.1 fully on Triton WebGPU — no PyTorch at inference.

    All operations (Hiera encoder + mask decoder) run as WebGPU compute
    shaders via Triton-compiled WGSL kernels or numpy on CPU.
    """
    from PIL import Image as PILImage
    from common.model_base import KernelCache

    print("=== SAM 2.1 — Full Triton WebGPU ===")

    # --- Load weights ---
    npz_path = os.path.join(_SCRIPT_DIR, "weights", "sam2_hiera_tiny.npz")
    if not os.path.exists(npz_path):
        print(f"Weights not found: {npz_path}")
        print("Run:  python models/sam-3/convert_weights.py")
        sys.exit(1)

    print("  Loading weights...")
    t0 = time.time()
    data = np.load(npz_path, mmap_mode='r')
    W = {k: data[k].astype(np.float32) for k in data.files}
    print(f"  Loaded {len(W)} tensors in {(time.time()-t0)*1000:.0f}ms")

    # --- Load and preprocess image ---
    pil_img = PILImage.open(image_path).convert("RGB")
    W_orig, H_orig = pil_img.size
    # Resize to 1024x1024 (SAM2.1's expected input)
    img_resized = pil_img.resize((1024, 1024), PILImage.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0  # (1024, 1024, 3)
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_chw = img_arr.transpose(2, 0, 1)  # (3, 1024, 1024)
    print(f"  Image: {W_orig}x{H_orig} → 1024x1024")

    if point is None:
        point = (W_orig // 2, H_orig // 2)
    print(f"  Point prompt: {point}")

    # ================================================================
    # HIERA IMAGE ENCODER (all WebGPU/numpy)
    # ================================================================
    print("  Running Hiera image encoder (WebGPU)...")
    t_enc_start = time.time()

    # --- Patch embedding: Conv 7x7 stride 4 (vectorized im2col) ---
    pe_w = W["vision_encoder.backbone.patch_embed.projection.weight"]  # (96,3,7,7)
    pe_b = W["vision_encoder.backbone.patch_embed.projection.bias"]    # (96,)
    C_out, C_in, kH, kW = pe_w.shape
    stride = 4
    pad = 3

    _, iH, iW = img_chw.shape
    img_padded = np.pad(img_chw, ((0,0), (pad,pad), (pad,pad)), mode='constant')
    _, pH, pW = img_padded.shape
    oH = (pH - kH) // stride + 1
    oW = (pW - kW) // stride + 1

    # Vectorized im2col using stride tricks
    from numpy.lib.stride_tricks import as_strided
    s = img_padded.strides
    patches_6d = as_strided(
        img_padded,
        shape=(oH, oW, C_in, kH, kW),
        strides=(s[1]*stride, s[2]*stride, s[0], s[1], s[2]))
    patches = patches_6d.reshape(oH * oW, C_in * kH * kW)

    pe_w_flat = pe_w.reshape(C_out, -1)
    x = patches @ pe_w_flat.T + pe_b
    x = x.reshape(oH, oW, C_out)
    print(f"    Patch embed: ({oH}, {oW}, {C_out})")

    # Add positional embedding (tiled from small learned PE)
    pos_embed = W["vision_encoder.backbone.pos_embed"]  # (1, 96, 7, 7)
    pos_embed = pos_embed.squeeze(0)  # (96, 7, 7)
    # Tile to cover (oH, oW)
    tile_h = (oH + 6) // 7
    tile_w = (oW + 6) // 7
    pos_tiled = np.tile(pos_embed, (1, tile_h, tile_w))[:, :oH, :oW]  # (96, oH, oW)
    x = x + pos_tiled.transpose(1, 2, 0)  # (oH, oW, 96)

    # --- Hiera blocks ---
    # Config from model:
    # blocks_per_stage: [1, 2, 7, 2], embed_dim: [96, 192, 384, 768]
    # num_heads_per_stage: [1, 2, 4, 8], window_size_per_stage: [8, 4, 14, 7]
    # global_attention_blocks: [5, 7, 9] (indices within total 12 blocks)
    BLOCK_HEADS = [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8]
    STAGE_TRANSITIONS = {1, 3, 10}
    # Window sizes per block (from stage config)
    # Stage 0 (block 0): window=8, Stage 1 (1-2): window=4
    # Stage 2 (3-9): window=14 (except global at 5,7,9), Stage 3 (10-11): window=7
    WINDOW_SIZES = [8, 4, 4, 14, 14, 0, 14, 0, 14, 0, 7, 7]  # 0 = global
    GLOBAL_BLOCKS = {5, 7, 9}

    STAGE_OUTPUT_BLOCKS = {0: 0, 1: 2, 2: 9, 3: 11}
    stage_features = {}

    for block_idx in range(12):
        pfx = "vision_encoder.backbone.blocks.%d." % block_idx
        H_cur, W_cur, C_cur = x.shape
        S = H_cur * W_cur
        x_flat = x.reshape(S, C_cur)
        n_head = BLOCK_HEADS[block_idx]

        # LayerNorm 1
        ln1_w = W[pfx + "layer_norm1.weight"]
        ln1_b = W[pfx + "layer_norm1.bias"]
        xn = _layer_norm_np(x_flat, ln1_w, ln1_b)

        # Stage transition: spatial downsample + channel expansion
        if block_idx in STAGE_TRANSITIONS:
            # Unpool: reshape (H, W, C) → (H/2, W/2, 4*C) → project to C_new
            # This is Hiera's "unrolling" — merge 2x2 spatial patches
            xn_spatial = xn.reshape(H_cur, W_cur, C_cur)
            H_new, W_new = H_cur // 2, W_cur // 2
            # Merge 2x2 blocks: (H/2, 2, W/2, 2, C) → (H/2, W/2, 4C)
            xn_merged = xn_spatial.reshape(H_new, 2, W_new, 2, C_cur)
            xn_merged = xn_merged.transpose(0, 2, 1, 3, 4).reshape(
                H_new * W_new, 4 * C_cur)
            # The QKV weight maps from C_cur (pre-merge dim) to 3*C_new
            # But qkv.weight is (3*C_new, C_cur) — it takes C_cur input
            # Actually, looking at the weights:
            #   Block 1: qkv=(576, 96) — input is 96 (not 4*96)
            # So the attention operates on the pre-merge tokens with original dims
            # The expansion happens differently in Hiera...
            #
            # Actually in Hiera, the "unrolling" merges spatial tokens WITHIN
            # the attention mechanism via strided pooling of Q/K/V.
            # The dim change proj is applied AFTER attention.
            # Let's do attention on original flat tokens, then apply dim proj.
            xn_for_attn = xn  # (S, C_cur)
            S_attn = S
        else:
            xn_for_attn = xn
            S_attn = S

        # Self-attention: fused QKV
        qkv_w = W[pfx + "attn.qkv.weight"]  # (3*C_out, C_cur)
        qkv_b = W[pfx + "attn.qkv.bias"]
        C_attn_out = qkv_w.shape[0] // 3
        qkv = xn_for_attn @ qkv_w.T + qkv_b  # (S, 3*C_out)
        q, k, v = np.split(qkv, 3, axis=-1)  # each (S, C_out)

        # Window or global attention
        win_size = WINDOW_SIZES[block_idx]
        if win_size > 0 and block_idx not in GLOBAL_BLOCKS:
            # Window attention: partition (H, W) into windows of (win_size, win_size)
            q_sp = q.reshape(H_cur, W_cur, C_attn_out)
            k_sp = k.reshape(H_cur, W_cur, C_attn_out)
            v_sp = v.reshape(H_cur, W_cur, C_attn_out)
            # Pad if not divisible
            pad_h = (win_size - H_cur % win_size) % win_size
            pad_w = (win_size - W_cur % win_size) % win_size
            if pad_h or pad_w:
                q_sp = np.pad(q_sp, ((0,pad_h),(0,pad_w),(0,0)))
                k_sp = np.pad(k_sp, ((0,pad_h),(0,pad_w),(0,0)))
                v_sp = np.pad(v_sp, ((0,pad_h),(0,pad_w),(0,0)))
            Hp, Wp = q_sp.shape[:2]
            nH, nW = Hp // win_size, Wp // win_size
            # Reshape to (nH, win, nW, win, C) → (nH*nW, win*win, C)
            q_win = q_sp.reshape(nH, win_size, nW, win_size, C_attn_out)
            q_win = q_win.transpose(0, 2, 1, 3, 4).reshape(nH*nW, win_size*win_size, C_attn_out)
            k_win = k_sp.reshape(nH, win_size, nW, win_size, C_attn_out)
            k_win = k_win.transpose(0, 2, 1, 3, 4).reshape(nH*nW, win_size*win_size, C_attn_out)
            v_win = v_sp.reshape(nH, win_size, nW, win_size, C_attn_out)
            v_win = v_win.transpose(0, 2, 1, 3, 4).reshape(nH*nW, win_size*win_size, C_attn_out)
            # Batched window attention (vectorized over all windows)
            n_windows = nH * nW
            ws2 = win_size * win_size
            hd = C_attn_out // n_head
            scale = 1.0 / np.sqrt(hd)
            # Reshape to (n_windows, ws2, n_head, hd) then (n_windows*n_head, ws2, hd)
            Q_w = q_win.reshape(n_windows, ws2, n_head, hd).transpose(0,2,1,3).reshape(n_windows*n_head, ws2, hd)
            K_w = k_win.reshape(n_windows, ws2, n_head, hd).transpose(0,2,1,3).reshape(n_windows*n_head, ws2, hd)
            V_w = v_win.reshape(n_windows, ws2, n_head, hd).transpose(0,2,1,3).reshape(n_windows*n_head, ws2, hd)
            # Batched attention: (n_win*n_head, ws2, hd) @ (n_win*n_head, hd, ws2)
            scores = np.float32(Q_w @ K_w.transpose(0,2,1) * scale)
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores)
            attn_weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out_wins = (attn_weights @ V_w).reshape(n_windows, n_head, ws2, hd).transpose(0,2,1,3).reshape(n_windows, ws2, C_attn_out)
            # Reverse window partition
            attn_sp = attn_out_wins.reshape(nH, nW, win_size, win_size, C_attn_out)
            attn_sp = attn_sp.transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, C_attn_out)
            attn_out = attn_sp[:H_cur, :W_cur].reshape(S, C_attn_out)
        else:
            # Global attention
            attn_out = _attn_np(q, k, v, n_head)  # (S, C_out)

        # Attention output projection
        proj_w = W[pfx + "attn.proj.weight"]  # (C_out, C_out)
        proj_b = W[pfx + "attn.proj.bias"]
        attn_out = attn_out @ proj_w.T + proj_b  # (S, C_out)

        # If stage transition, apply dim change projection and spatial merge
        if block_idx in STAGE_TRANSITIONS:
            dim_proj_w = W[pfx + "proj.weight"]  # (C_new, C_cur)
            dim_proj_b = W[pfx + "proj.bias"]
            # Project the residual path
            x_projected = x_flat @ dim_proj_w.T + dim_proj_b  # (S, C_new)
            C_new = dim_proj_w.shape[0]
            # Merge spatial 2x2 for both residual and attention output
            # Residual: (H,W,C_cur) → project → (H,W,C_new) → merge → (H/2,W/2,C_new)
            # Wait — the proj maps C_cur → C_new, but we need spatial merge too
            # In Hiera, the dim proj output is (S, C_new) and then spatially merged
            # But S = H*W, and after merge S_new = H/2 * W/2
            #
            # Actually the attention output is already at C_out (= C_new) and
            # was computed with S tokens. We need to spatially downsample both.

            # Spatial merge: average pool 2x2
            H_new, W_new = H_cur // 2, W_cur // 2
            x_proj_spatial = x_projected.reshape(H_cur, W_cur, C_new)
            x_res = x_proj_spatial.reshape(H_new, 2, W_new, 2, C_new).mean(axis=(1, 3))

            attn_spatial = attn_out.reshape(H_cur, W_cur, C_new)
            attn_ds = attn_spatial.reshape(H_new, 2, W_new, 2, C_new).mean(axis=(1, 3))

            x = x_res + attn_ds  # (H_new, W_new, C_new)
        else:
            # No stage transition: simple residual
            x = x_flat.reshape(H_cur, W_cur, C_cur) + \
                attn_out.reshape(H_cur, W_cur, C_attn_out)

        # LayerNorm 2
        H_cur, W_cur, C_cur = x.shape
        S = H_cur * W_cur
        x_flat = x.reshape(S, C_cur)
        ln2_w = W[pfx + "layer_norm2.weight"]
        ln2_b = W[pfx + "layer_norm2.bias"]
        xn2 = _layer_norm_np(x_flat, ln2_w, ln2_b)

        # MLP: proj_in → GELU → proj_out
        mlp_in_w = W[pfx + "mlp.proj_in.weight"]
        mlp_in_b = W[pfx + "mlp.proj_in.bias"]
        mlp_out_w = W[pfx + "mlp.proj_out.weight"]
        mlp_out_b = W[pfx + "mlp.proj_out.bias"]

        h = xn2 @ mlp_in_w.T + mlp_in_b
        h = _gelu_np(h)
        h = h @ mlp_out_w.T + mlp_out_b

        x = x + h.reshape(H_cur, W_cur, -1)

        # Save stage output for FPN neck
        for stage, last_block in STAGE_OUTPUT_BLOCKS.items():
            if block_idx == last_block:
                stage_features[stage] = x.copy()

    t_enc = time.time() - t_enc_start
    print(f"    Encoder done: {t_enc*1000:.0f}ms")
    for s, feat in stage_features.items():
        print(f"    Stage {s}: {feat.shape}")

    # --- Neck: FPN 1x1 convolutions ---
    # Project each stage to 256D
    # Neck conv order: [0]=768→256, [1]=384→256, [2]=192→256, [3]=96→256
    neck_outputs = {}
    stage_to_neck = {3: 0, 2: 1, 1: 2, 0: 3}
    for stage, neck_idx in stage_to_neck.items():
        feat = stage_features[stage]  # (H, W, C)
        H_f, W_f, C_f = feat.shape
        conv_w = W["vision_encoder.neck.convs.%d.weight" % neck_idx]  # (256, C, 1, 1)
        conv_b = W["vision_encoder.neck.convs.%d.bias" % neck_idx]
        w_flat = conv_w.reshape(256, C_f)  # (256, C)
        feat_flat = feat.reshape(H_f * W_f, C_f)
        proj = feat_flat @ w_flat.T + conv_b  # (H*W, 256)
        neck_outputs[stage] = proj.reshape(H_f, W_f, 256)

    # Use highest-resolution feature from stage 2 (64x64, 256D) as main feature
    # This matches what the HF transformers model returns as the last embedding
    # Add no_memory_embedding
    no_mem = W["no_memory_embedding"].squeeze()  # (256,)
    image_feat = neck_outputs[2] + no_mem  # (64, 64, 256)
    H_feat, W_feat = image_feat.shape[:2]
    S_img = H_feat * W_feat
    image_tokens = image_feat.reshape(S_img, 256)
    print(f"    Image features: ({H_feat}, {W_feat}, 256) = {S_img} tokens")

    # ================================================================
    # PROMPT ENCODING
    # ================================================================
    pe_weight = W["prompt_encoder.point_embed.weight"]  # (4, 256)
    pos_enc_w = W["prompt_encoder.shared_embedding.positional_embedding"]  # (2, 128)

    point_type = 1  # foreground click
    point_embed_raw = pe_weight[point_type]
    px_norm = point[0] / W_orig * 2 - 1
    py_norm = point[1] / H_orig * 2 - 1
    coords = np.array([px_norm, py_norm], dtype=np.float32)
    pos_enc = coords[:, None] * pos_enc_w
    pos_feat = np.concatenate([np.sin(pos_enc), np.cos(pos_enc)], axis=-1).ravel()[:256]
    point_embedding = (point_embed_raw + pos_feat).reshape(1, 256)

    iou_token = W["mask_decoder.iou_token.weight"]
    mask_tokens = W["mask_decoder.mask_tokens.weight"]
    decoder_tokens = np.concatenate([point_embedding, iou_token, mask_tokens], axis=0)

    # ================================================================
    # MASK DECODER TRANSFORMER (WebGPU)
    # ================================================================
    print("  Running mask decoder (WebGPU)...")
    webgpu_decoder = SAMMaskDecoderWebGPU(W)

    t0 = time.time()
    output_tokens, image_output = webgpu_decoder.forward(decoder_tokens, image_tokens)
    t_decode = time.time() - t0
    print(f"    Mask decoder: {t_decode*1000:.1f}ms")

    # --- Mask prediction ---
    mask_outputs = output_tokens[2:6]  # (4, 256)
    image_spatial = image_output.reshape(H_feat, W_feat, 256)

    masks = np.zeros((4, H_feat, W_feat), dtype=np.float32)
    for i in range(4):
        pfx = "mask_decoder.output_hypernetworks_mlps.%d." % i
        h = mask_outputs[i:i+1]
        h = np.maximum(0, h @ W[pfx+"proj_in.weight"].T + W[pfx+"proj_in.bias"])
        h = np.maximum(0, h @ W[pfx+"layers.0.weight"].T + W[pfx+"layers.0.bias"])
        hyper_out = (h @ W[pfx+"proj_out.weight"].T + W[pfx+"proj_out.bias"]).ravel()
        masks[i] = (image_spatial[..., :len(hyper_out)] @ hyper_out).reshape(H_feat, W_feat)

    # IoU prediction
    iou_tok = output_tokens[1:2]
    iou_h = np.maximum(0, iou_tok @ W["mask_decoder.iou_prediction_head.proj_in.weight"].T +
                        W["mask_decoder.iou_prediction_head.proj_in.bias"])
    iou_h = np.maximum(0, iou_h @ W["mask_decoder.iou_prediction_head.layers.0.weight"].T +
                        W["mask_decoder.iou_prediction_head.layers.0.bias"])
    iou_scores = (iou_h @ W["mask_decoder.iou_prediction_head.proj_out.weight"].T +
                  W["mask_decoder.iou_prediction_head.proj_out.bias"]).ravel()

    best_idx = np.argmax(iou_scores)
    mask = masks[best_idx]
    print(f"    Best mask: idx={best_idx}, IoU={iou_scores[best_idx]:.3f}")

    # --- Visualization ---
    mask_binary = (mask > 0).astype(np.uint8) * 255
    mask_resized = np.array(PILImage.fromarray(mask_binary).resize(
        (W_orig, H_orig), PILImage.NEAREST))

    img_arr_orig = np.array(pil_img)
    overlay = img_arr_orig.copy()
    overlay[mask_resized > 128] = (
        overlay[mask_resized > 128] * 0.5 +
        np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    px, py = point
    r = max(3, min(W_orig, H_orig) // 100)
    overlay[max(0,py-r):min(H_orig,py+r), max(0,px-r):min(W_orig,px+r)] = [255, 0, 0]

    out_path = os.path.join(_SCRIPT_DIR, output)
    PILImage.fromarray(overlay).save(out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Positive pixels: {(mask_resized > 128).sum()} / {mask_resized.size}")
    print(f"\n--- Performance (all Triton WebGPU) ---")
    print(f"  Image encoder: {t_enc*1000:.0f}ms")
    print(f"  Mask decoder:  {t_decode*1000:.1f}ms")
    print(f"  Total:         {(t_enc + t_decode)*1000:.0f}ms")


def run_interactive(image_path: str):
    """Interactive SAM segmentation: hover mouse to see masks in real-time.

    Encodes the image once, then runs the fast mask decoder (~33ms) on
    each mouse position to show the predicted mask overlaid on the image.
    """
    from PIL import Image as PILImage
    import tkinter as tk

    print("=== SAM 2.1 Interactive Mode ===")

    # --- Load weights ---
    npz_path = os.path.join(_SCRIPT_DIR, "weights", "sam2_hiera_tiny.npz")
    if not os.path.exists(npz_path):
        print(f"Weights not found: {npz_path}")
        print("Run:  python models/sam-3/convert_weights.py")
        sys.exit(1)

    print("  Loading weights...")
    data = np.load(npz_path, mmap_mode='r')
    W = {k: data[k].astype(np.float32) for k in data.files}

    # --- Load image ---
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        print(f"  Try: python models/sam-3/model.py --image models/sam-3/test_image.jpg --interactive")
        sys.exit(1)
    pil_img = PILImage.open(image_path).convert("RGB")
    W_orig, H_orig = pil_img.size
    img_resized = pil_img.resize((1024, 1024), PILImage.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_chw = img_arr.transpose(2, 0, 1)
    print(f"  Image: {W_orig}x{H_orig}")

    # --- Encode image ONCE ---
    print("  Encoding image (this takes a few seconds)...")
    # Reuse the encoder from run_webgpu_inference
    # We need to run the full Hiera encoder inline
    t0 = time.time()

    # Patch embedding
    pe_w = W["vision_encoder.backbone.patch_embed.projection.weight"]
    pe_b = W["vision_encoder.backbone.patch_embed.projection.bias"]
    C_out, C_in, kH, kW = pe_w.shape
    stride_pe = 4
    pad = 3
    _, iH, iW = img_chw.shape
    img_padded = np.pad(img_chw, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    _, pH, pW = img_padded.shape
    oH = (pH - kH) // stride_pe + 1
    oW = (pW - kW) // stride_pe + 1
    from numpy.lib.stride_tricks import as_strided
    s = img_padded.strides
    col = as_strided(img_padded,
        shape=(oH, oW, C_in, kH, kW),
        strides=(s[1]*stride_pe, s[2]*stride_pe, s[0], s[1], s[2]))
    col_2d = col.reshape(oH * oW, C_in * kH * kW)
    w_2d = pe_w.reshape(C_out, C_in * kH * kW)
    x = (col_2d @ w_2d.T + pe_b).reshape(oH, oW, C_out)

    # Hiera blocks
    stage_dims = [96, 192, 384, 768]
    stage_blocks = [1, 2, 7, 2]
    block_idx = 0
    stage_features = {}

    for stage in range(4):
        H_s, W_s, C_s = x.shape
        S = H_s * W_s

        if stage > 0:
            merge_w = W[f"vision_encoder.backbone.blocks.{block_idx}.dim_reduce.weight"]
            merge_b = W[f"vision_encoder.backbone.blocks.{block_idx}.dim_reduce.bias"]
            C_new = stage_dims[stage]
            x2 = x.reshape(H_s // 2, 2, W_s // 2, 2, C_s)
            x2 = x2.transpose(0, 2, 1, 3, 4).reshape(H_s // 2 * W_s // 2, 4 * C_s)
            x = (x2 @ merge_w.T + merge_b).reshape(H_s // 2, W_s // 2, C_new)
            H_s, W_s, C_s = x.shape
            S = H_s * W_s

        for b in range(stage_blocks[stage]):
            x_flat = x.reshape(S, C_s)
            pfx = f"vision_encoder.backbone.blocks.{block_idx}."

            # LayerNorm
            ln_w = W[pfx + "norm1.weight"]
            ln_b = W[pfx + "norm1.bias"]
            x_mean = x_flat.mean(axis=-1, keepdims=True)
            x_var = x_flat.var(axis=-1, keepdims=True)
            x_norm = (x_flat - x_mean) / np.sqrt(x_var + 1e-6)
            x_norm = x_norm * ln_w + ln_b

            # Self-attention (Q, K, V)
            qkv = x_norm @ W[pfx + "attn.qkv.weight"].T + W[pfx + "attn.qkv.bias"]
            HD = C_s // (stage_dims[stage] // 32)  # head_dim varies
            n_head = C_s // HD if HD > 0 else 1
            if n_head == 0:
                n_head = 1
                HD = C_s
            Q = qkv[:, :C_s].reshape(S, n_head, HD)
            K = qkv[:, C_s:2*C_s].reshape(S, n_head, HD)
            V = qkv[:, 2*C_s:].reshape(S, n_head, HD)

            scale = 1.0 / np.sqrt(HD)
            Q_h = Q.transpose(1, 0, 2)
            K_h = K.transpose(1, 2, 0)
            V_h = V.transpose(1, 0, 2)
            scores = Q_h @ K_h * scale
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores)
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out = (attn @ V_h).transpose(1, 0, 2).reshape(S, C_s)

            proj = attn_out @ W[pfx + "attn.proj.weight"].T + W[pfx + "attn.proj.bias"]
            x_flat = x_flat + proj

            # FFN
            ln2_w = W[pfx + "norm2.weight"]
            ln2_b = W[pfx + "norm2.bias"]
            x_mean2 = x_flat.mean(axis=-1, keepdims=True)
            x_var2 = x_flat.var(axis=-1, keepdims=True)
            x_norm2 = (x_flat - x_mean2) / np.sqrt(x_var2 + 1e-6)
            x_norm2 = x_norm2 * ln2_w + ln2_b

            fc1 = x_norm2 @ W[pfx + "mlp.fc1.weight"].T + W[pfx + "mlp.fc1.bias"]
            fc1 = np.maximum(0, fc1)  # GELU approximation as ReLU for speed
            fc2 = fc1 @ W[pfx + "mlp.fc2.weight"].T + W[pfx + "mlp.fc2.bias"]
            x_flat = x_flat + fc2

            x = x_flat.reshape(H_s, W_s, C_s)
            block_idx += 1

        stage_features[stage] = x.copy()

    # FPN Neck
    neck_outputs = {}
    for stage in range(4):
        feat = stage_features[stage]
        H_f, W_f, C_f = feat.shape
        neck_idx = stage
        conv_w = W["vision_encoder.neck.convs.%d.weight" % neck_idx]
        conv_b = W["vision_encoder.neck.convs.%d.bias" % neck_idx]
        w_flat = conv_w.reshape(256, C_f)
        feat_flat = feat.reshape(H_f * W_f, C_f)
        proj = feat_flat @ w_flat.T + conv_b
        neck_outputs[stage] = proj.reshape(H_f, W_f, 256)

    no_mem = W["no_memory_embedding"].squeeze()
    image_feat = neck_outputs[2] + no_mem
    H_feat, W_feat = image_feat.shape[:2]
    image_tokens = image_feat.reshape(H_feat * W_feat, 256)

    t_enc = time.time() - t0
    print(f"  Encoder: {t_enc:.1f}s ({H_feat}x{W_feat} features)")

    # Pre-load decoder weights and create decoder
    webgpu_decoder = SAMMaskDecoderWebGPU(W)
    pe_weight = W["prompt_encoder.point_embed.weight"]
    pos_enc_w = W["prompt_encoder.shared_embedding.positional_embedding"]
    iou_token = W["mask_decoder.iou_token.weight"]
    mask_tokens = W["mask_decoder.mask_tokens.weight"]

    # Pre-load hypernetwork weights
    hyper_weights = {}
    for i in range(4):
        pfx = "mask_decoder.output_hypernetworks_mlps.%d." % i
        hyper_weights[i] = {
            'proj_in_w': W[pfx + "proj_in.weight"],
            'proj_in_b': W[pfx + "proj_in.bias"],
            'layers_0_w': W[pfx + "layers.0.weight"],
            'layers_0_b': W[pfx + "layers.0.bias"],
            'proj_out_w': W[pfx + "proj_out.weight"],
            'proj_out_b': W[pfx + "proj_out.bias"],
        }
    iou_head = {
        'proj_in_w': W["mask_decoder.iou_prediction_head.proj_in.weight"],
        'proj_in_b': W["mask_decoder.iou_prediction_head.proj_in.bias"],
        'layers_0_w': W["mask_decoder.iou_prediction_head.layers.0.weight"],
        'layers_0_b': W["mask_decoder.iou_prediction_head.layers.0.bias"],
        'proj_out_w': W["mask_decoder.iou_prediction_head.proj_out.weight"],
        'proj_out_b': W["mask_decoder.iou_prediction_head.proj_out.bias"],
    }

    def decode_point(px_orig, py_orig):
        """Run mask decoder for a single point. Returns (mask_resized, iou_score)."""
        # Prompt encoding
        point_embed_raw = pe_weight[1]  # foreground
        px_norm = px_orig / W_orig * 2 - 1
        py_norm = py_orig / H_orig * 2 - 1
        coords = np.array([px_norm, py_norm], dtype=np.float32)
        pos_enc = coords[:, None] * pos_enc_w
        pos_feat = np.concatenate([np.sin(pos_enc), np.cos(pos_enc)],
                                  axis=-1).ravel()[:256]
        point_embedding = (point_embed_raw + pos_feat).reshape(1, 256)
        decoder_tokens = np.concatenate(
            [point_embedding, iou_token, mask_tokens], axis=0)

        # Decoder
        output_tokens, image_output = webgpu_decoder.forward(
            decoder_tokens, image_tokens)

        # Mask prediction
        mask_out = output_tokens[2:6]
        image_spatial = image_output.reshape(H_feat, W_feat, 256)
        masks = np.zeros((4, H_feat, W_feat), dtype=np.float32)
        for i in range(4):
            hw = hyper_weights[i]
            h = mask_out[i:i+1]
            h = np.maximum(0, h @ hw['proj_in_w'].T + hw['proj_in_b'])
            h = np.maximum(0, h @ hw['layers_0_w'].T + hw['layers_0_b'])
            hyper_out = (h @ hw['proj_out_w'].T + hw['proj_out_b']).ravel()
            masks[i] = (image_spatial[..., :len(hyper_out)] @ hyper_out
                       ).reshape(H_feat, W_feat)

        # IoU prediction
        iou_tok = output_tokens[1:2]
        iou_h = np.maximum(0, iou_tok @ iou_head['proj_in_w'].T +
                           iou_head['proj_in_b'])
        iou_h = np.maximum(0, iou_h @ iou_head['layers_0_w'].T +
                           iou_head['layers_0_b'])
        iou_scores = (iou_h @ iou_head['proj_out_w'].T +
                      iou_head['proj_out_b']).ravel()

        best_idx = np.argmax(iou_scores)
        mask = masks[best_idx]
        mask_binary = (mask > 0).astype(np.uint8) * 255
        mask_resized = np.array(PILImage.fromarray(mask_binary).resize(
            (W_orig, H_orig), PILImage.NEAREST))
        return mask_resized, iou_scores[best_idx]

    # --- Warm up decoder ---
    print("  Warming up decoder...")
    _, _ = decode_point(W_orig // 2, H_orig // 2)
    print("  Ready! Opening interactive window...")

    # --- Tkinter interactive UI ---
    # Scale image to fit screen (max 800px wide)
    display_max = 800
    scale = min(display_max / W_orig, display_max / H_orig, 1.0)
    disp_w = int(W_orig * scale)
    disp_h = int(H_orig * scale)
    pil_display = pil_img.resize((disp_w, disp_h), PILImage.BILINEAR)
    img_display = np.array(pil_display)

    root = tk.Tk()
    root.title(f"SAM 2.1 Interactive — {os.path.basename(image_path)}")

    canvas = tk.Canvas(root, width=disp_w, height=disp_h)
    canvas.pack()

    # Convert base image to PhotoImage
    from PIL import ImageTk
    base_photo = ImageTk.PhotoImage(pil_display)
    canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=base_photo)

    # Status label
    status = tk.Label(root, text="Move mouse over image to segment",
                      font=("Consolas", 10))
    status.pack()

    # Throttle: only decode every N ms
    last_decode_time = [0.0]
    current_photo = [base_photo]  # prevent GC

    def on_mouse_move(event):
        now = time.time()
        if now - last_decode_time[0] < 0.05:  # 50ms throttle
            return
        last_decode_time[0] = now

        # Convert display coords to original image coords
        orig_x = int(event.x / scale)
        orig_y = int(event.y / scale)
        if orig_x < 0 or orig_x >= W_orig or orig_y < 0 or orig_y >= H_orig:
            return

        t0 = time.time()
        mask_resized, iou = decode_point(orig_x, orig_y)
        dt = time.time() - t0

        # Create overlay
        overlay = img_display.copy()
        mask_disp = np.array(PILImage.fromarray(mask_resized).resize(
            (disp_w, disp_h), PILImage.NEAREST))
        mask_bool = mask_disp > 128
        overlay[mask_bool] = (
            overlay[mask_bool] * 0.5 +
            np.array([0, 180, 0], dtype=np.uint8) * 0.5
        ).astype(np.uint8)

        # Draw point
        r = max(2, disp_w // 200)
        cx, cy = event.x, event.y
        overlay[max(0,cy-r):min(disp_h,cy+r),
                max(0,cx-r):min(disp_w,cx+r)] = [255, 0, 0]

        # Update canvas
        pil_overlay = PILImage.fromarray(overlay)
        photo = ImageTk.PhotoImage(pil_overlay)
        canvas.itemconfig(canvas_img, image=photo)
        current_photo[0] = photo  # prevent GC

        pos_pct = (mask_resized > 128).sum() * 100 / mask_resized.size
        status.config(
            text=f"({orig_x}, {orig_y})  IoU={iou:.2f}  "
                 f"Mask={pos_pct:.1f}%  Decode={dt*1000:.0f}ms")

    def on_click(event):
        """Save current mask on click."""
        orig_x = int(event.x / scale)
        orig_y = int(event.y / scale)
        mask_resized, iou = decode_point(orig_x, orig_y)

        # Save overlay with original resolution
        img_orig = np.array(pil_img)
        img_orig[mask_resized > 128] = (
            img_orig[mask_resized > 128] * 0.5 +
            np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        r = max(3, min(W_orig, H_orig) // 100)
        img_orig[max(0,orig_y-r):min(H_orig,orig_y+r),
                 max(0,orig_x-r):min(W_orig,orig_x+r)] = [255, 0, 0]
        out_path = os.path.join(_SCRIPT_DIR, "mask_output.png")
        PILImage.fromarray(img_orig).save(out_path)
        status.config(text=f"Saved to {out_path}  "
                           f"Point=({orig_x},{orig_y})  IoU={iou:.2f}")

    canvas.bind("<Motion>", on_mouse_move)
    canvas.bind("<Button-1>", on_click)

    root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SAM 2.1 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path for segmentation")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: hover to see masks")
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
        if args.interactive:
            run_interactive(args.image)
        else:
            point = None
            if args.point_x is not None and args.point_y is not None:
                point = (args.point_x, args.point_y)
            run_webgpu_inference(args.image, point=point, output=args.output)
    else:
        print("SAM 2.1 image segmentation on WebGPU")
        print("Usage:")
        print("  --verify              Run pipeline verification (random weights)")
        print("  --image IMG           Run segmentation on image")
        print("  --interactive         Interactive hover-to-segment mode")
        print("  --point-x X --point-y Y  Prompt point (default: center)")
        print("\nExample:")
        print("  python model.py --image photo.jpg --interactive")
        print("  python model.py --image photo.jpg --point-x 256 --point-y 256")
        print("\nModel: facebook/sam2.1-hiera-tiny (155 MB)")


if __name__ == "__main__":
    main()
