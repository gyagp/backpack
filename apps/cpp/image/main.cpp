/**
 * Backpack Image — Text-to-image generation using bp::Session.
 *
 * Supports Z-Image-Turbo pipeline:
 *   Text Encoder (Qwen3 4B) → DiT Transformer → VAE Decoder → Image
 *
 * All pipeline orchestration is in this app. The runtime (backpack.dll)
 * only provides Device, Model, Tensor, Session — it doesn't know about
 * "diffusion" or "image generation".
 *
 * Usage:
 *   backpack_image --model path/to/z-image-turbo/webgpu/ \
 *       --prompt "A cat sitting on a rainbow" --output cat.ppm
 */

#include "backpack.h"

// Internal headers for tokenizer (shared with LLM app)
#include "onnx_tokenizer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── PPM image writer ───────────────────────────────────────────────────────

static void writePPM(const std::string& path, const float* rgb,
                      int width, int height) {
    // rgb is [3, H, W] in CHW order, values in [0, 1]
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * height * width + y * width + x];
                v = std::max(0.0f, std::min(1.0f, v));
                f.put((char)(int)(v * 255.0f + 0.5f));
            }
        }
    }
    printf("  Saved: %s (%dx%d)\n", path.c_str(), width, height);
}

// ─── Scheduler: compute timestep schedule ───────────────────────────────────
// Z-Image-Turbo uses a flow-matching schedule with few steps (typically 4).

struct Schedule {
    std::vector<float> sigmas;    // noise levels at each step
    std::vector<float> timesteps; // timestep values for the model
};

static Schedule computeSchedule(int numSteps) {
    Schedule s;
    // Linear schedule from 1.0 to 0.0
    for (int i = 0; i <= numSteps; i++) {
        s.sigmas.push_back(1.0f - (float)i / numSteps);
    }
    // Timesteps = sigmas * 1000
    for (int i = 0; i < numSteps; i++) {
        s.timesteps.push_back(s.sigmas[i] * 1000.0f);
    }
    return s;
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string modelDir, prompt, outputPath = "output.ppm";
    std::string backendStr;
    int width = 512, height = 512;
    int numSteps = 4;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)     modelDir = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--output" && i+1 < argc) outputPath = argv[++i];
        else if (arg == "--width" && i+1 < argc)  width = atoi(argv[++i]);
        else if (arg == "--height" && i+1 < argc) height = atoi(argv[++i]);
        else if (arg == "--steps" && i+1 < argc)  numSteps = atoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc)   seed = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backendStr = argv[++i];
    }

    if (modelDir.empty() || prompt.empty()) {
        fprintf(stderr,
            "Backpack Image — WebGPU text-to-image generation\n\n"
            "Usage: %s --model <dir> --prompt <text> [options]\n\n"
            "  --output <path>   Output image (PPM format, default: output.ppm)\n"
            "  --width <n>       Image width (default: 512, must be multiple of 8)\n"
            "  --height <n>      Image height (default: 512, must be multiple of 8)\n"
            "  --steps <n>       Denoising steps (default: 4)\n"
            "  --seed <n>        Random seed (default: 42)\n"
            "  --backend <name>  vulkan / d3d12 / metal\n", argv[0]);
        return 1;
    }

    int latentH = height / 8;
    int latentW = width / 8;
    printf("Backpack Image: %dx%d (%d steps, seed %d)\n", width, height, numSteps, seed);
    printf("Prompt: \"%s\"\n", prompt.c_str());

    // ─── 1. Create device ────────────────────────────────────────────────

    bp::Backend backend = bp::Backend::Default;
    if (backendStr == "d3d12")       backend = bp::Backend::D3D12;
    else if (backendStr == "vulkan") backend = bp::Backend::Vulkan;
    else if (backendStr == "metal")  backend = bp::Backend::Metal;

    auto device = bp::Device::Create(backend);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }
    printf("GPU: %s (%s)\n\n", device.GetName().c_str(),
           device.GetBackendName().c_str());

    // ─── 2. Load models ──────────────────────────────────────────────────

    auto t0 = std::chrono::steady_clock::now();

    // ─── 3. Tokenize prompt ──────────────────────────────────────────────

    OnnxTokenizer tokenizer;
    tokenizer.load(modelDir);

    auto inputIds = tokenizer.encode(prompt);
    int seqLen = (int)inputIds.size();
    printf("Prompt: %d tokens\n", seqLen);

    // Pad to max 512 for the text encoder
    int maxSeqLen = 512;
    std::vector<int64_t> inputIds64(maxSeqLen, 0);
    std::vector<int64_t> attentionMask(maxSeqLen, 0);
    for (int i = 0; i < seqLen && i < maxSeqLen; i++) {
        inputIds64[i] = inputIds[i];
        attentionMask[i] = 1;
    }

    // ─── 4. Run text encoder ─────────────────────────────────────────────

    printf("\n--- Text Encoder ---\n"); fflush(stdout);
    auto tEnc0 = std::chrono::steady_clock::now();

    auto textEnc = bp::Model::Load(device,
        (fs::path(modelDir) / "text_encoder_model_q4f16.onnx").string());

    auto inputIdsTensor = bp::Tensor::Create(device, bp::DataType::Int64,
                                              {1, (int64_t)maxSeqLen});
    inputIdsTensor.SetData(inputIds64.data(), maxSeqLen * sizeof(int64_t));

    auto attMaskTensor = bp::Tensor::Create(device, bp::DataType::Int64,
                                             {1, (int64_t)maxSeqLen});
    attMaskTensor.SetData(attentionMask.data(), maxSeqLen * sizeof(int64_t));

    auto hiddenTensor = bp::Tensor::Create(device, bp::DataType::Float32,
                                           {1, (int64_t)maxSeqLen, 2560});

    auto encSession = bp::Session::Create(textEnc);
    encSession.SetInput("input_ids", inputIdsTensor);
    encSession.SetInput("attention_mask", attMaskTensor);
    encSession.SetOutput("encoder_hidden_state", hiddenTensor);
    encSession.Run();

    // Free text encoder to save VRAM (~1.5 GB)
    encSession.Release();
    textEnc.Release();
    inputIdsTensor.Release();
    attMaskTensor.Release();

    auto tEnc1 = std::chrono::steady_clock::now();
    auto encMs = std::chrono::duration<double,std::milli>(tEnc1 - tEnc0).count();
    printf("  Text encoder: %.0fms\n", encMs);

    // Debug: check encoder hidden state — read first 10 values
    {
        int64_t nHidden = hiddenTensor.GetElementCount();
        int readN = std::min((int)nHidden, 10);
        std::vector<float> hData(readN);
        hiddenTensor.GetData(hData.data(), readN * sizeof(float));
        printf("  Encoder[0:10] = ");
        for (int i = 0; i < readN; i++) printf("%.4f ", hData[i]);
        printf("\n");
    }

    // ─── 5. Initialize latents ───────────────────────────────────────────

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int latentSize = 16 * latentH * latentW;
    std::vector<float> latentsData(latentSize);
    for (auto& v : latentsData) v = dist(rng);

    auto latentsTensor = bp::Tensor::Create(device, bp::DataType::Float32,
                                             {1, 16, 1, (int64_t)latentH, (int64_t)latentW});
    latentsTensor.SetData(latentsData.data(), latentSize * sizeof(float));

    // ─── 6. Diffusion loop ───────────────────────────────────────────────
    // Load DiT and scheduler for the diffusion steps

    printf("\n--- Diffusion (%d steps) ---\n", numSteps);
    auto schedule = computeSchedule(numSteps);

    printf("Loading transformer (DiT)...\n"); fflush(stdout);
    auto dit = bp::Model::Load(device,
        (fs::path(modelDir) / "transformer_model_q4f16.onnx").string());

    printf("Loading scheduler...\n"); fflush(stdout);
    auto scheduler = bp::Model::Load(device,
        (fs::path(modelDir) / "scheduler_step_model_f16.onnx").string());

    for (int step = 0; step < numSteps; step++) {
        auto tStep0 = std::chrono::steady_clock::now();

        // 6a. Run DiT transformer
        float timestep = schedule.timesteps[step];
        auto timestepTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
        timestepTensor.SetData(&timestep, sizeof(float));

        auto noisePredTensor = bp::Tensor::Create(device, bp::DataType::Float32,
            {16, 1, (int64_t)latentH, (int64_t)latentW});

        // Debug: check latent input
        {
            int readN = 10;
            std::vector<float> lData(readN);
            latentsTensor.GetData(lData.data(), readN * sizeof(float));
            printf("  Latent in[0:10] = ");
            for (int i = 0; i < readN; i++) printf("%.4f ", lData[i]);
            printf("\n");
        }

        auto ditSession = bp::Session::Create(dit);
        ditSession.SetInput("hidden_states", latentsTensor);
        ditSession.SetInput("timestep", timestepTensor);
        ditSession.SetInput("encoder_hidden_states", hiddenTensor);
        ditSession.SetOutput("unified_results", noisePredTensor);
        ditSession.Run();

        // Debug: check DiT output
        {
            int64_t nPred = noisePredTensor.GetElementCount();
            int readN = std::min((int)nPred, 10);
            std::vector<float> predData(readN);
            noisePredTensor.GetData(predData.data(), readN * sizeof(float));
            printf("  DiT out[0:10] = ");
            for (int i = 0; i < readN; i++) printf("%.4f ", predData[i]);
            printf("\n");
        }

        // 6b. Run scheduler step
        float stepInfo[2] = { schedule.sigmas[step], schedule.sigmas[step + 1] };
        auto stepInfoTensor = bp::Tensor::Create(device, bp::DataType::Float32, {2});
        stepInfoTensor.SetData(stepInfo, 2 * sizeof(float));

        auto latentsOutTensor = bp::Tensor::Create(device, bp::DataType::Float32,
            {1, 16, 1, (int64_t)latentH, (int64_t)latentW});

        auto schedSession = bp::Session::Create(scheduler);
        schedSession.SetInput("noise_pred", noisePredTensor);
        schedSession.SetInput("latents", latentsTensor);
        schedSession.SetInput("step_info", stepInfoTensor);
        schedSession.SetOutput("latents_out", latentsOutTensor);
        schedSession.Run();

        latentsTensor = std::move(latentsOutTensor);

        auto tStep1 = std::chrono::steady_clock::now();
        auto stepMs = std::chrono::duration<double,std::milli>(tStep1 - tStep0).count();
        printf("  Step %d/%d: %.0fms (t=%.1f)\n", step+1, numSteps, stepMs, timestep);
        fflush(stdout);
    }

    // Note: don't release dit/scheduler — destructor cleanup can hang.
    // Memory will be freed when the process exits.
    // dit.Release();
    // scheduler.Release();
    // hiddenTensor.Release();

    // ─── 7. VAE decode ───────────────────────────────────────────────────

    printf("\n--- VAE Decode ---\n"); fflush(stdout);
    auto tVae0 = std::chrono::steady_clock::now();

    auto imageTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, 3, (int64_t)height, (int64_t)width});

    printf("Loading VAE...\n"); fflush(stdout);
    auto vaePre = bp::Model::Load(device,
        (fs::path(modelDir) / "vae_pre_process_model_f16.onnx").string());
    auto vae = bp::Model::Load(device,
        (fs::path(modelDir) / "vae_decoder_model_f16.onnx").string());

    // Preprocess: [1,16,1,H,W] → [1,16,H,W]
    auto scaledTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, 16, (int64_t)latentH, (int64_t)latentW});

    auto preSession = bp::Session::Create(vaePre);
    preSession.SetInput("latents", latentsTensor);
    preSession.SetOutput("scaled_latents", scaledTensor);
    preSession.Run();

    // VAE decode: [1,16,H,W] → [1,3,H*8,W*8]
    auto vaeSession = bp::Session::Create(vae);
    vaeSession.SetInput("latent_sample", scaledTensor);
    vaeSession.SetOutput("sample", imageTensor);
    vaeSession.Run();

    auto tVae1 = std::chrono::steady_clock::now();
    auto vaeMs = std::chrono::duration<double,std::milli>(tVae1 - tVae0).count();
    printf("  VAE decode: %.0fms\n", vaeMs);

    // ─── 8. Save image ───────────────────────────────────────────────────

    int imageSize = 3 * height * width;
    std::vector<float> imageData(imageSize);
    imageTensor.GetData(imageData.data(), imageSize * sizeof(float));

    // Debug: check raw pixel statistics
    float minV = 1e30f, maxV = -1e30f;
    double sumV = 0;
    int nanCount = 0;
    for (int i = 0; i < imageSize; i++) {
        float v = imageData[i];
        if (std::isnan(v) || std::isinf(v)) { nanCount++; continue; }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sumV += v;
    }
    printf("  Raw pixels: min=%.4f max=%.4f avg=%.4f nan=%d/%d\n",
           minV, maxV, sumV / (imageSize - nanCount), nanCount, imageSize);
    fflush(stdout);

    writePPM(outputPath, imageData.data(), width, height);

    // ─── Summary ─────────────────────────────────────────────────────────

    auto tEnd = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration<double,std::milli>(tEnd - t0).count();
    printf("\n--- Performance ---\n");
    printf("  Total:    %.0fms\n", totalMs);
    printf("  Encode:   %.0fms\n", encMs);
    printf("  Diffusion: %d steps\n", numSteps);
    printf("  VAE:      %.0fms\n", vaeMs);
    printf("  Output:   %s (%dx%d)\n", outputPath.c_str(), width, height);

    return 0;
}
