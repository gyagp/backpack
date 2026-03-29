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
#include "gpu_context.h"

// Internal headers for tokenizer (shared with LLM app)
#include "onnx_tokenizer.h"

// Shared app utilities
#include "../common/app_common.h"

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
    // rgb is [3, H, W] in CHW order, values in [-1, 1]
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * height * width + y * width + x];
                v = v * 0.5f + 0.5f;
                v = std::max(0.0f, std::min(1.0f, v));
                f.put((char)(int)(v * 255.0f + 0.5f));
            }
        }
    }
    printf("  Saved: %s (%dx%d)\n", path.c_str(), width, height);
}

static void printTensorStats(const char* label, const std::vector<float>& data) {
    if (data.empty()) {
        printf("  %s: empty\n", label);
        return;
    }
    float minV = 1e30f, maxV = -1e30f;
    double sumV = 0.0;
    int nanCount = 0;
    for (float v : data) {
        if (std::isnan(v) || std::isinf(v)) {
            nanCount++;
            continue;
        }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sumV += v;
    }
    int validCount = (int)data.size() - nanCount;
    double avg = validCount > 0 ? (sumV / validCount) : 0.0;
    printf("  %s: min=%.4f max=%.4f avg=%.4f nan=%d/%zu\n",
           label, minV, maxV, avg, nanCount, data.size());
}

// app::fp16ToFloat: use app::app::fp16ToFloat from app_common.h

static void printRuntimeTensorStats(const char* label, bp::Tensor& tensor) {
    auto shape = tensor.GetShape();
    int64_t count = 1;
    for (auto d : shape) count *= d;
    if (count <= 0) count = 1;
    std::vector<float> values((size_t)count);
    if (tensor.GetDtype() == bp::DataType::Float32) {
        tensor.GetData(values.data(), values.size() * sizeof(float));
    } else {
        std::vector<uint16_t> fp16((size_t)count);
        tensor.GetData(fp16.data(), fp16.size() * sizeof(uint16_t));
        for (int64_t i = 0; i < count; i++) values[(size_t)i] = app::fp16ToFloat(fp16[(size_t)i]);
    }
    printTensorStats(label, values);
}

static constexpr bool kDebugProbes = true;
static constexpr bool kDebugHiddenStateProbes = false;

// ─── Scheduler: compute timestep schedule ───────────────────────────────────
// Z-Image-Turbo uses a flow-matching schedule with few steps (typically 4).

struct Schedule {
    std::vector<float> sigmas;    // noise levels at each step
    std::vector<float> timesteps; // raw scheduler timesteps in [0, 1000]
};

static Schedule computeSchedule(int numSteps) {
    Schedule s;
    // Linear schedule from 1.0 to 0.0
    for (int i = 0; i <= numSteps; i++) {
        s.sigmas.push_back(1.0f - (float)i / numSteps);
    }
    // Flow-match scheduler timesteps before the model-side normalization.
    for (int i = 0; i < numSteps; i++) {
        s.timesteps.push_back(s.sigmas[i] * 1000.0f);
    }
    return s;
}

static float transformerTimestepFromSchedule(const Schedule& schedule, int step) {
    // Z-Image uses the transform: (1000 - t) / 1000 = 1 - sigma.
    // The ONNX model multiplies by 1000 internally, so passing (1 - sigma)
    // gives internal timestep = (1 - sigma) * 1000 = 1000 - t.
    // At step 0 (full noise, sigma=1.0), this gives 0.0 → internal 0.
    return 1.0f - schedule.timesteps[step] / 1000.0f;
}

static float schedulerStepCoeff(float sigmaCur, float sigmaNext) {
    // Match scheduler_step_model_f16.onnx algebra, but evaluate in fp32 to
    // avoid step-to-step drift from the model's tiny fp16 recurrence.
    const float shift = 3.0f;
    const float mul = 1000.0f;
    const float clone = 1.0f;
    const float sub1 = -997.0f;
    const float sub2 = 2.0f;
    const float zero = 0.0f;

    const float sub = sigmaNext - clone;
    const float denom = (sub == zero) ? clone : sub;
    const float div = sub1 / denom;
    const float div1 = (mul + sigmaCur * div) / mul;
    const float div2 = (mul + (sigmaCur + clone) * div) / mul;
    const float div3 = (shift * div1) / (sub2 * div1 + clone);
    const float div4 = (shift * div2) / (sub2 * div2 + clone);
    const float where1 = (sigmaCur >= sub) ? zero : div4;
    return where1 - div3;
}

static void runSchedulerStepStable(const std::vector<float>& noisePred,
                                   const std::vector<float>& latents,
                                   float sigmaCur,
                                   float sigmaNext,
                                   std::vector<float>& latentsOut) {
    const float coeff = schedulerStepCoeff(sigmaCur, sigmaNext);
    if (latentsOut.size() != latents.size()) latentsOut.resize(latents.size());
    for (size_t i = 0; i < latents.size(); i++) {
        latentsOut[i] = latents[i] - coeff * noisePred[i];
    }
}

static void scaleTransformerInput(const std::vector<float>& latents,
                                  float sigma,
                                  std::vector<float>& scaledLatents) {
    float denom = std::sqrt(sigma * sigma + 1.0f);
    if (denom == 0.0f) denom = 1.0f;
    if (scaledLatents.size() != latents.size()) scaledLatents.resize(latents.size());
    for (size_t i = 0; i < latents.size(); i++) {
        scaledLatents[i] = latents[i] / denom;
    }
}

// Chat template: use app::applyChatTemplate from app_common.h

// ─── CLI ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string modelDir, prompt, outputPath = "output.ppm";
    std::string backendStr;
    int width = 512, height = 512;
    int numSteps = 4;
    int seed = 42;
    float guidanceScale = 5.0f;
    bool reloadTransformerPerStep = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)     modelDir = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--output" && i+1 < argc) outputPath = argv[++i];
        else if (arg == "--width" && i+1 < argc)  width = atoi(argv[++i]);
        else if (arg == "--height" && i+1 < argc) height = atoi(argv[++i]);
        else if (arg == "--steps" && i+1 < argc)  numSteps = atoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc)   seed = atoi(argv[++i]);
        else if (arg == "--guidance-scale" && i+1 < argc) guidanceScale = (float)atof(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backendStr = argv[++i];
        else if (arg == "--reload-transformer-per-step") reloadTransformerPerStep = true;
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
            "  --guidance-scale <f>  Classifier-free guidance scale (default: 5.0)\n"
            "  --backend <name>  vulkan / d3d12 / metal\n"
            "  --reload-transformer-per-step  Deprecated no-op; multi-step correctness no longer requires reload\n", argv[0]);
        return 1;
    }

    int latentH = height / 8;
    int latentW = width / 8;
    printf("Backpack Image: %dx%d (%d steps, seed %d)\n", width, height, numSteps, seed);
    printf("Prompt: \"%s\"\n", prompt.c_str());
    if (reloadTransformerPerStep) {
        printf("Note: --reload-transformer-per-step is deprecated and ignored; using stable multi-step path.\n");
        reloadTransformerPerStep = false;
    }

    // ─── 1. Create device ────────────────────────────────────────────────

    auto device = app::createDevice(backendStr);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }
    app::printGpuInfo(device);
    printf("\n");

    // ─── 2. Load models ──────────────────────────────────────────────────

    auto t0 = std::chrono::steady_clock::now();

    // ─── 3. Tokenize prompt ──────────────────────────────────────────────

    OnnxTokenizer tokenizer;
    tokenizer.load(modelDir);

    int maxSeqLen = 512;

    // ─── 4. Run text encoder ─────────────────────────────────────────────

    printf("\n--- Text Encoder ---\n"); fflush(stdout);
    auto tEnc0 = std::chrono::steady_clock::now();

    auto textEnc = bp::Model::Load(device,
        (fs::path(modelDir) / "text_encoder_model_q4f16.onnx").string());

    auto encodePromptHidden = [&](const std::string& promptText,
                                  bool printDebugStats) -> std::pair<bp::Tensor, int> {
        std::string chatPrompt = app::applyChatTemplate(promptText, "");
        auto inputIds = tokenizer.encode(chatPrompt);
        int seqLen = (int)inputIds.size();

        std::vector<int64_t> inputIds64(maxSeqLen, 0);
        std::vector<int64_t> attentionMask(maxSeqLen, 0);
        for (int i = 0; i < seqLen && i < maxSeqLen; i++) {
            inputIds64[i] = inputIds[i];
            attentionMask[i] = 1;
        }

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
        encSession.Release();

        const int tokenCount = std::min(seqLen, maxSeqLen);
        std::vector<float> hiddenDebug((size_t)maxSeqLen * 2560);
        hiddenTensor.GetData(hiddenDebug.data(), hiddenDebug.size() * sizeof(float));
        if (printDebugStats) {
            printf("Prompt: %d tokens\n", seqLen);
            printTensorStats("Encoder hidden", hiddenDebug);
        }

        std::vector<float> trimmedHidden((size_t)tokenCount * 2560);
        for (int token = 0; token < tokenCount; token++) {
            memcpy(trimmedHidden.data() + (size_t)token * 2560,
                   hiddenDebug.data() + (size_t)token * 2560,
                   2560 * sizeof(float));
        }

        auto hiddenTrimTensor = bp::Tensor::Create(device, bp::DataType::Float32,
                                                   {1, (int64_t)tokenCount, 2560});
        hiddenTrimTensor.SetData(trimmedHidden.data(), trimmedHidden.size() * sizeof(float));

        hiddenTensor.Release();
        inputIdsTensor.Release();
        attMaskTensor.Release();
        return {std::move(hiddenTrimTensor), tokenCount};
    };

    auto [hiddenTrimTensor, promptTokenCount] = encodePromptHidden(prompt, true);

    // Save encoder hidden states for Python ORT comparison
    {
        auto dumpDir = fs::path(outputPath).parent_path();
        std::vector<float> encDump((size_t)promptTokenCount * 2560);
        hiddenTrimTensor.GetData(encDump.data(), encDump.size() * sizeof(float));
        auto dumpEnc = dumpDir / "debug_encoder_hidden.bin";
        std::ofstream f(dumpEnc, std::ios::binary);
        f.write((char*)encDump.data(), encDump.size() * sizeof(float));
        printf("  Saved encoder hidden: %s (%d tokens x 2560)\n",
               dumpEnc.string().c_str(), promptTokenCount);
    }

    auto [negativeHiddenTrimTensor, negativePromptTokenCount] =
        encodePromptHidden("", guidanceScale > 1.0f);

    // Save negative encoder hidden states for Python ORT comparison
    if (guidanceScale > 1.0f) {
        auto dumpDir = fs::path(outputPath).parent_path();
        std::vector<float> negDump((size_t)negativePromptTokenCount * 2560);
        negativeHiddenTrimTensor.GetData(negDump.data(), negDump.size() * sizeof(float));
        auto dumpNeg = dumpDir / "debug_neg_encoder_hidden.bin";
        std::ofstream f(dumpNeg, std::ios::binary);
        f.write((char*)negDump.data(), negDump.size() * sizeof(float));
        printf("  Saved neg encoder hidden: %s (%d tokens)\n",
               dumpNeg.string().c_str(), negativePromptTokenCount);
    }

    auto padHiddenTensorToLength = [&](bp::Tensor& srcTensor,
                                       int srcTokenCount,
                                       int targetTokenCount) -> bp::Tensor {
        if (srcTokenCount >= targetTokenCount) return std::move(srcTensor);

        std::vector<float> srcHidden((size_t)srcTokenCount * 2560);
        srcTensor.GetData(srcHidden.data(), srcHidden.size() * sizeof(float));

        std::vector<float> paddedHidden((size_t)targetTokenCount * 2560, 0.0f);
        for (int token = 0; token < srcTokenCount; token++) {
            memcpy(paddedHidden.data() + (size_t)token * 2560,
                   srcHidden.data() + (size_t)token * 2560,
                   2560 * sizeof(float));
        }

        auto paddedTensor = bp::Tensor::Create(device, bp::DataType::Float32,
                                               {1, (int64_t)targetTokenCount, 2560});
        paddedTensor.SetData(paddedHidden.data(), paddedHidden.size() * sizeof(float));
        srcTensor.Release();
        return paddedTensor;
    };

    if (guidanceScale > 1.0f) {
        int cfgTokenCount = std::max(promptTokenCount, negativePromptTokenCount);
        hiddenTrimTensor = padHiddenTensorToLength(hiddenTrimTensor, promptTokenCount, cfgTokenCount);
        negativeHiddenTrimTensor = padHiddenTensorToLength(negativeHiddenTrimTensor, negativePromptTokenCount, cfgTokenCount);
        promptTokenCount = cfgTokenCount;
        negativePromptTokenCount = cfgTokenCount;
    }

    if (guidanceScale > 1.0f) {
        printf("  Negative prompt: %d tokens (CFG scale %.2f)\n",
               negativePromptTokenCount, guidanceScale);
    }

    textEnc.Release();
    // Force GPU memory cleanup before loading next model
    {
        auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
        gpuCtx->waitForQueue();
        gpuCtx->flushBufferPool();
    }

    auto tEnc1 = std::chrono::steady_clock::now();
    auto encMs = std::chrono::duration<double,std::milli>(tEnc1 - tEnc0).count();
    printf("  Text encoder: %.0fms\n", encMs);


    // ─── 5. Initialize latents ───────────────────────────────────────────

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int latentSize = 16 * latentH * latentW;
    std::vector<float> latentsData(latentSize);
    for (auto& v : latentsData) v = dist(rng);
    printTensorStats("Initial latents", latentsData);

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

    // Pre-allocate reusable tensors to avoid per-step GPU buffer allocation
    auto timestepTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
    auto modelLatentsTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, 16, 1, (int64_t)latentH, (int64_t)latentW});
    auto noisePredTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {16, 1, (int64_t)latentH, (int64_t)latentW});
    auto noisePredUncondTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {16, 1, (int64_t)latentH, (int64_t)latentW});
    auto noisePredPreCastTensor = bp::Tensor::Create(device, bp::DataType::Float16,
        {16, 1, (int64_t)latentH, (int64_t)latentW});
    auto layer0Add3Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto whereTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto unsqueeze21Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner1Add4Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto contextRefiner1Add1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto concat18Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0Add2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0AttentionNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0Mul5Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0FfnNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0Mul6Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0W1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0SigmoidTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0MulTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0Mul1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0W2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto layer0FfnNorm2PreCastTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer0NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer8NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer16NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer20NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer22NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23Add3Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23AttentionNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23AttentionNorm2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23TanhTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23AttentionOutTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23Mul5Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer23FfnNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24NormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24Add3Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24AttentionNorm2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24TanhTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24AttentionOutTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24Mul5Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24FfnNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24W2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer24NormPreCastTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer28FfnNorm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer28W2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredLayer28NormPreCastTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredFinalNormTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredFinalMatMulTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredFinalAddTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredSplitTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredSqueezeTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredSliceTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredReshape13Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noisePredTranspose2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto hiddenStatesCastTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto hiddenStatesGatherTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto hiddenStatesReshape6Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    // Noise refiner internal probes
    auto noiseRefiner0Norm1Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0Mul5Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0Add3Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0FfnNorm2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0Add4Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0MhaQTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1, 256, 3840});
    auto noiseRefiner0Add2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0TanhTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0AttnNorm2Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto noiseRefiner0AttnOutTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto latentsOutTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, 16, 1, (int64_t)latentH, (int64_t)latentW});
    auto xEmbedderOutputTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
    std::vector<float> debugNoisePred(latentSize);
    std::vector<float> debugNoisePredUncond(latentSize);
    std::vector<float> debugLatents(latentSize);
    std::vector<float> currentLatents = latentsData;
    std::vector<float> modelLatents(latentSize);

    for (int step = 0; step < numSteps; step++) {
        auto tStep0 = std::chrono::steady_clock::now();

        if (reloadTransformerPerStep && step > 0) {
            dit.Release();
            dit = bp::Model::Load(device,
                (fs::path(modelDir) / "transformer_model_q4f16.onnx").string());
        }

        // 6a. Run DiT transformer
        float timestep = transformerTimestepFromSchedule(schedule, step);
        timestepTensor.SetData(&timestep, sizeof(float));
        scaleTransformerInput(currentLatents, schedule.sigmas[step], modelLatents);
        modelLatentsTensor.SetData(modelLatents.data(), latentSize * sizeof(float));

        auto runTransformer = [&](bp::Tensor& promptHiddenTensor,
                                  bp::Tensor& outputTensor,
                                  bool enableDebugOutputs) {
            auto ditSession = bp::Session::Create(dit);
            ditSession.SetInput("hidden_states", modelLatentsTensor);
            ditSession.SetInput("timestep", timestepTensor);
            ditSession.SetInput("encoder_hidden_states", promptHiddenTensor);
            if (step == 0) {
                ditSession.SetOutput("/2-1/Gemm_output_0", xEmbedderOutputTensor);
            }
            if (kDebugHiddenStateProbes && step == 0) {
                ditSession.SetOutput("graph_input_cast_0", hiddenStatesCastTensor);
                ditSession.SetOutput("/Gather_output_0", hiddenStatesGatherTensor);
                ditSession.SetOutput("/Reshape_6_output_0", hiddenStatesReshape6Tensor);
            }
            if (enableDebugOutputs && kDebugProbes) {
                ditSession.SetOutput("/Where_output_0", whereTensor);
                ditSession.SetOutput("/Unsqueeze_21_output_0", unsqueeze21Tensor);
                ditSession.SetOutput("/noise_refiner.0/attention_norm1/norm_out", noiseRefiner0Norm1Tensor);
                ditSession.SetOutput("/noise_refiner.0/Mul_5_output_0", noiseRefiner0Mul5Tensor);
                ditSession.SetOutput("/noise_refiner.0/Add_3_output_0", noiseRefiner0Add3Tensor);
                ditSession.SetOutput("/noise_refiner.0/Add_2_output_0", noiseRefiner0Add2Tensor);
                ditSession.SetOutput("/noise_refiner.0/Tanh_output_0", noiseRefiner0TanhTensor);
                ditSession.SetOutput("/noise_refiner.0/attention_norm2/norm_out", noiseRefiner0AttnNorm2Tensor);
                ditSession.SetOutput("/noise_refiner.0/attention/to_out.0/MatMul_quant_output_cast_0", noiseRefiner0AttnOutTensor);
                ditSession.SetOutput("/noise_refiner.0/ffn_norm2/norm_out", noiseRefiner0FfnNorm2Tensor);
                ditSession.SetOutput("/noise_refiner.0/Add_4_output_0", noiseRefiner0Add4Tensor);
                ditSession.SetOutput("/noise_refiner.0/attention/mha/mha_q_in", noiseRefiner0MhaQTensor);
                ditSession.SetOutput("/noise_refiner.1/Add_4_output_0", noiseRefiner1Add4Tensor);
                ditSession.SetOutput("/context_refiner.1/Add_1_output_0", contextRefiner1Add1Tensor);
                ditSession.SetOutput("/Concat_18_output_0", concat18Tensor);
                ditSession.SetOutput("/layers.0/Add_3_output_0", layer0Add3Tensor);
                ditSession.SetOutput("/layers.0/Add_2_output_0", layer0Add2Tensor);
                ditSession.SetOutput("/layers.0/attention_norm1/norm_out", layer0AttentionNorm1Tensor);
                ditSession.SetOutput("/layers.0/Mul_5_output_0", layer0Mul5Tensor);
                ditSession.SetOutput("/layers.0/ffn_norm1/norm_out", layer0FfnNorm1Tensor);
                ditSession.SetOutput("/layers.0/Mul_6_output_0", layer0Mul6Tensor);
                ditSession.SetOutput("/layers.0/feed_forward/w1/MatMul_output_0", layer0W1Tensor);
                ditSession.SetOutput("/layers.0/feed_forward/Sigmoid_output_0", layer0SigmoidTensor);
                ditSession.SetOutput("/layers.0/feed_forward/Mul_output_0", layer0MulTensor);
                ditSession.SetOutput("/layers.0/feed_forward/Mul_1_output_cast_0", layer0Mul1Tensor);
                ditSession.SetOutput("/layers.0/feed_forward/w2/MatMul_quant_output_cast_0", layer0W2Tensor);
                ditSession.SetOutput("/layers.0/ffn_norm2/SimplifiedLayerNormalization_output_cast_0", layer0FfnNorm2PreCastTensor);
                ditSession.SetOutput("/layers.0/ffn_norm2/norm_out", noisePredLayer0NormTensor);
                ditSession.SetOutput("/layers.8/ffn_norm2/norm_out", noisePredLayer8NormTensor);
                ditSession.SetOutput("/layers.16/ffn_norm2/norm_out", noisePredLayer16NormTensor);
                ditSession.SetOutput("/layers.20/ffn_norm2/norm_out", noisePredLayer20NormTensor);
                ditSession.SetOutput("/layers.22/ffn_norm2/norm_out", noisePredLayer22NormTensor);
                ditSession.SetOutput("/layers.23/ffn_norm2/norm_out", noisePredLayer23NormTensor);
                ditSession.SetOutput("/layers.23/Add_3_output_0", noisePredLayer23Add3Tensor);
                ditSession.SetOutput("/layers.23/attention_norm1/norm_out", noisePredLayer23AttentionNorm1Tensor);
                ditSession.SetOutput("/layers.23/attention_norm2/norm_out", noisePredLayer23AttentionNorm2Tensor);
                ditSession.SetOutput("/layers.23/Tanh_output_0", noisePredLayer23TanhTensor);
                ditSession.SetOutput("/layers.23/attention/to_out.0/MatMul_output_0", noisePredLayer23AttentionOutTensor);
                ditSession.SetOutput("/layers.23/Mul_5_output_0", noisePredLayer23Mul5Tensor);
                ditSession.SetOutput("/layers.23/ffn_norm1/norm_out", noisePredLayer23FfnNorm1Tensor);
                ditSession.SetOutput("/layers.24/ffn_norm2/norm_out", noisePredLayer24NormTensor);
                ditSession.SetOutput("/layers.24/Add_3_output_0", noisePredLayer24Add3Tensor);
                ditSession.SetOutput("/layers.24/attention_norm2/norm_out", noisePredLayer24AttentionNorm2Tensor);
                ditSession.SetOutput("/layers.24/Tanh_output_0", noisePredLayer24TanhTensor);
                ditSession.SetOutput("/layers.24/attention/to_out.0/MatMul_output_0", noisePredLayer24AttentionOutTensor);
                ditSession.SetOutput("/layers.24/Mul_5_output_0", noisePredLayer24Mul5Tensor);
                ditSession.SetOutput("/layers.24/ffn_norm1/norm_out", noisePredLayer24FfnNorm1Tensor);
                ditSession.SetOutput("/layers.24/feed_forward/w2/MatMul_quant_output_cast_0", noisePredLayer24W2Tensor);
                ditSession.SetOutput("/layers.24/ffn_norm2/SimplifiedLayerNormalization_output_cast_0", noisePredLayer24NormPreCastTensor);
                ditSession.SetOutput("/layers.28/ffn_norm1/norm_out", noisePredLayer28FfnNorm1Tensor);
                ditSession.SetOutput("/layers.28/feed_forward/w2/MatMul_quant_output_cast_0", noisePredLayer28W2Tensor);
                ditSession.SetOutput("/layers.28/ffn_norm2/SimplifiedLayerNormalization_output_cast_0", noisePredLayer28NormPreCastTensor);
                ditSession.SetOutput("/layers.28/ffn_norm2/norm_out", noisePredFinalNormTensor);
                ditSession.SetOutput("/2-1/linear/MatMul_output_0", noisePredFinalMatMulTensor);
                ditSession.SetOutput("/2-1/linear/Add_output_0", noisePredFinalAddTensor);
                ditSession.SetOutput("/Split_output_0", noisePredSplitTensor);
                ditSession.SetOutput("/Squeeze_output_0", noisePredSqueezeTensor);
                ditSession.SetOutput("/Slice_4_output_0", noisePredSliceTensor);
                ditSession.SetOutput("/Reshape_13_output_0", noisePredReshape13Tensor);
                ditSession.SetOutput("/Transpose_2_output_0", noisePredTranspose2Tensor);
                ditSession.SetOutput("graph_output_cast_0", noisePredPreCastTensor);
            }
            ditSession.SetOutput("unified_results", outputTensor);
            ditSession.Run();
        };

        runTransformer(hiddenTrimTensor, noisePredTensor, true);

        // Save raw transformer output for step 0
        if (step == 0) {
            printRuntimeTensorStats("x_embedder Gemm", xEmbedderOutputTensor);
            std::vector<float> rawNoisePred(latentSize);
            noisePredTensor.GetData(rawNoisePred.data(), latentSize * sizeof(float));
            auto dumpDir = fs::path(outputPath).parent_path();
            auto dumpNoise = dumpDir / "debug_noise_pred_raw.bin";
            std::ofstream f(dumpNoise, std::ios::binary);
            f.write((char*)rawNoisePred.data(), rawNoisePred.size() * sizeof(float));
            printf("  Saved raw noise pred: %s (%zu floats)\n",
                   dumpNoise.string().c_str(), rawNoisePred.size());
        }

        if (kDebugHiddenStateProbes && step == 0) {
            printRuntimeTensorStats("Hidden cast step 1", hiddenStatesCastTensor);
            printRuntimeTensorStats("Hidden gather step 1", hiddenStatesGatherTensor);
            printRuntimeTensorStats("Hidden reshape6 step 1", hiddenStatesReshape6Tensor);
        }
        if (guidanceScale > 1.0f) {
            runTransformer(negativeHiddenTrimTensor, noisePredUncondTensor, false);
            noisePredTensor.GetData(debugNoisePred.data(), latentSize * sizeof(float));
            noisePredUncondTensor.GetData(debugNoisePredUncond.data(), latentSize * sizeof(float));

            // Debug: print cond and uncond stats before CFG
            printTensorStats("Cond noise pred", debugNoisePred);
            printTensorStats("Uncond noise pred", debugNoisePredUncond);

            for (int i = 0; i < latentSize; i++) {
                debugNoisePred[i] = debugNoisePredUncond[i] +
                    guidanceScale * (debugNoisePred[i] - debugNoisePredUncond[i]);
            }
            noisePredTensor.SetData(debugNoisePred.data(), latentSize * sizeof(float));
        }

        if (kDebugProbes) {
            char whereLabel[64];
            snprintf(whereLabel, sizeof(whereLabel), "Where step %d", step + 1);
            printRuntimeTensorStats(whereLabel, whereTensor);

            char unsqueeze21Label[64];
            snprintf(unsqueeze21Label, sizeof(unsqueeze21Label), "Unsqueeze21 step %d", step + 1);
            printRuntimeTensorStats(unsqueeze21Label, unsqueeze21Tensor);

            // Noise refiner.0 internal probes
            printRuntimeTensorStats("NR0 attn_norm1", noiseRefiner0Norm1Tensor);
            printRuntimeTensorStats("NR0 Mul5(gate*attn)", noiseRefiner0Mul5Tensor);
            printRuntimeTensorStats("NR0 Add2(adaLN)", noiseRefiner0Add2Tensor);
            printRuntimeTensorStats("NR0 Tanh(gate1)", noiseRefiner0TanhTensor);
            printRuntimeTensorStats("NR0 attn_out", noiseRefiner0AttnOutTensor);
            printRuntimeTensorStats("NR0 attn_norm2", noiseRefiner0AttnNorm2Tensor);
            printRuntimeTensorStats("NR0 MHA Q", noiseRefiner0MhaQTensor);
            printRuntimeTensorStats("NR0 Add3(+attn)", noiseRefiner0Add3Tensor);
            printRuntimeTensorStats("NR0 ffn_norm2", noiseRefiner0FfnNorm2Tensor);
            printRuntimeTensorStats("NR0 Add4(+ffn)", noiseRefiner0Add4Tensor);

            // Dump NR0 MHA Q input for comparison with ORT
            {
                auto shape = noiseRefiner0MhaQTensor.GetShape();
                int64_t count = 1;
                for (auto d : shape) count *= d;
                if (count > 0) {
                    std::vector<float> qData((size_t)count);
                    noiseRefiner0MhaQTensor.GetData(qData.data(), qData.size() * sizeof(float));
                    auto dumpDir = fs::path(outputPath).parent_path();
                    auto dumpQ = dumpDir / "debug_nr0_mha_q.bin";
                    std::ofstream f(dumpQ, std::ios::binary);
                    f.write((char*)qData.data(), qData.size() * sizeof(float));
                    printf("  Saved NR0 MHA Q: %s (%lld floats, shape=[",
                           dumpQ.string().c_str(), (long long)count);
                    for (size_t i = 0; i < shape.size(); i++)
                        printf("%s%lld", i?",":"", (long long)shape[i]);
                    printf("])\n");
                }
            }

            char noiseRefinerLabel[64];
            snprintf(noiseRefinerLabel, sizeof(noiseRefinerLabel), "Noise refiner step %d", step + 1);
            printRuntimeTensorStats(noiseRefinerLabel, noiseRefiner1Add4Tensor);

            char contextRefinerLabel[64];
            snprintf(contextRefinerLabel, sizeof(contextRefinerLabel), "Context refiner step %d", step + 1);
            printRuntimeTensorStats(contextRefinerLabel, contextRefiner1Add1Tensor);

            char concat18Label[64];
            snprintf(concat18Label, sizeof(concat18Label), "Concat18 step %d", step + 1);
            printRuntimeTensorStats(concat18Label, concat18Tensor);

            std::vector<float> debugNoisePreCast(latentSize);
            if (noisePredPreCastTensor.GetDtype() == bp::DataType::Float32) {
                noisePredPreCastTensor.GetData(debugNoisePreCast.data(), latentSize * sizeof(float));
            } else {
                std::vector<uint16_t> debugNoisePreCastFp16(latentSize);
                noisePredPreCastTensor.GetData(debugNoisePreCastFp16.data(), latentSize * sizeof(uint16_t));
                for (int i = 0; i < latentSize; i++) debugNoisePreCast[i] = app::fp16ToFloat(debugNoisePreCastFp16[i]);
            }
            char noisePreLabel[64];
            snprintf(noisePreLabel, sizeof(noisePreLabel), "Noise pre-cast step %d", step + 1);
            printTensorStats(noisePreLabel, debugNoisePreCast);

            // Dump the Slice_4 output (pre-unpatchification tokens) for comparison with ORT
            if (step == 0) {
                auto sliceShape = noisePredSliceTensor.GetShape();
                int64_t sliceCount = 1;
                for (auto d : sliceShape) sliceCount *= d;
                if (sliceCount > 0 && sliceCount <= 65536) {
                    std::vector<float> sliceData;
                    if (noisePredSliceTensor.GetDtype() == bp::DataType::Float32) {
                        sliceData.resize((size_t)sliceCount);
                        noisePredSliceTensor.GetData(sliceData.data(), sliceData.size() * sizeof(float));
                    } else {
                        std::vector<uint16_t> fp16Data((size_t)sliceCount);
                        noisePredSliceTensor.GetData(fp16Data.data(), fp16Data.size() * sizeof(uint16_t));
                        sliceData.resize((size_t)sliceCount);
                        for (int64_t i = 0; i < sliceCount; i++) sliceData[(size_t)i] = app::fp16ToFloat(fp16Data[(size_t)i]);
                    }
                    auto dumpDir = fs::path(outputPath).parent_path();
                    auto dumpSlice = dumpDir / "debug_slice4.bin";
                    std::ofstream f(dumpSlice, std::ios::binary);
                    f.write((char*)sliceData.data(), sliceData.size() * sizeof(float));
                    printf("  Saved Slice_4: %s (%lld floats, shape=[",
                           dumpSlice.string().c_str(), (long long)sliceCount);
                    for (size_t i = 0; i < sliceShape.size(); i++)
                        printf("%s%lld", i?",":"", (long long)sliceShape[i]);
                    printf("])\n");
                    printTensorStats("Slice_4", sliceData);
                }
            }
        }

        if (guidanceScale <= 1.0f) {
            noisePredTensor.GetData(debugNoisePred.data(), latentSize * sizeof(float));
        }
        char noiseLabel[64];
        snprintf(noiseLabel, sizeof(noiseLabel), "Noise pred step %d", step + 1);
        printTensorStats(noiseLabel, debugNoisePred);

        // Evaluate the tiny scheduler recurrence in fp32 for every denoise step.
        // The exported ONNX scheduler uses fp16 internal casts and becomes
        // numerically unstable once the denoiser path is corrected.
        runSchedulerStepStable(debugNoisePred, currentLatents,
                               schedule.sigmas[step], schedule.sigmas[step + 1],
                               debugLatents);
        latentsOutTensor.SetData(debugLatents.data(), latentSize * sizeof(float));
        char latentLabel[64];
        snprintf(latentLabel, sizeof(latentLabel), "Latents step %d", step + 1);
        printTensorStats(latentLabel, debugLatents);

        // Swap: output becomes input for next step
        currentLatents.swap(debugLatents);
        std::swap(latentsTensor, latentsOutTensor);

        auto tStep1 = std::chrono::steady_clock::now();
        auto stepMs = std::chrono::duration<double,std::milli>(tStep1 - tStep0).count();
         printf("  Step %d/%d: %.0fms (t=%.4f, sigma=%.4f)\n",
             step+1, numSteps, stepMs, timestep, schedule.sigmas[step]);
        fflush(stdout);
    }

    // Note: don't release dit — destructor cleanup can hang.
    // Memory will be freed when the process exits.
    // dit.Release();
    // hiddenTensor.Release();

    // ─── 7. VAE decode ───────────────────────────────────────────────────

    printf("\n--- VAE Decode ---\n"); fflush(stdout);
    auto tVae0 = std::chrono::steady_clock::now();

    auto imageTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, 3, (int64_t)height, (int64_t)width});
    auto vaeInputTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
    auto vaeLatentsFp16Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto vaeConvInTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto vaeConvNormAddTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto vaeConvActMulTensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});
    auto vaeSampleFp16Tensor = bp::Tensor::Create(device, bp::DataType::Float16, {1});

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

    std::vector<float> scaledData(latentSize);
    scaledTensor.GetData(scaledData.data(), latentSize * sizeof(float));
    printTensorStats("Scaled latents", scaledData);
    if (kDebugProbes) printRuntimeTensorStats("Scaled latents runtime", scaledTensor);

    // Save scaled latents and initial latents for Python ORT comparison
    {
        auto dumpDir = fs::path(outputPath).parent_path();
        auto dumpScaled = dumpDir / "debug_scaled_latents.bin";
        auto dumpInitial = dumpDir / "debug_initial_latents.bin";
        std::ofstream f1(dumpScaled, std::ios::binary);
        f1.write((char*)scaledData.data(), scaledData.size() * sizeof(float));
        printf("  Saved scaled latents: %s (%zu floats)\n", dumpScaled.string().c_str(), scaledData.size());
        std::ofstream f2(dumpInitial, std::ios::binary);
        f2.write((char*)latentsData.data(), latentsData.size() * sizeof(float));
        printf("  Saved initial latents: %s (%zu floats)\n", dumpInitial.string().c_str(), latentsData.size());
    }

    // VAE decode: [1,16,H,W] → [1,3,H*8,W*8]
    auto vaeSession = bp::Session::Create(vae);
    vaeSession.SetInput("latent_sample", scaledTensor);
    if (kDebugProbes) {
        vaeSession.SetOutput("latent_sample", vaeInputTensor);
        vaeSession.SetOutput("latent_sample_to_fp16", vaeLatentsFp16Tensor);
        vaeSession.SetOutput("/decoder/conv_in/Conv_output_0", vaeConvInTensor);
        vaeSession.SetOutput("/decoder/conv_norm_out/Add_output_0", vaeConvNormAddTensor);
        vaeSession.SetOutput("/decoder/conv_act/Mul_output_0", vaeConvActMulTensor);
        vaeSession.SetOutput("sample_fp16", vaeSampleFp16Tensor);
    }
    vaeSession.SetOutput("sample", imageTensor);
    vaeSession.Run();

    auto tVae1 = std::chrono::steady_clock::now();
    auto vaeMs = std::chrono::duration<double,std::milli>(tVae1 - tVae0).count();
    printf("  VAE decode: %.0fms\n", vaeMs);

    if (kDebugProbes) {
        printRuntimeTensorStats("VAE input", vaeInputTensor);
        printRuntimeTensorStats("VAE latent_fp16", vaeLatentsFp16Tensor);
        printRuntimeTensorStats("VAE conv_in", vaeConvInTensor);
        printRuntimeTensorStats("VAE conv_norm_add", vaeConvNormAddTensor);
        printRuntimeTensorStats("VAE conv_act_mul", vaeConvActMulTensor);
        printRuntimeTensorStats("VAE sample_fp16", vaeSampleFp16Tensor);
    }

    // ─── 8. Save image ───────────────────────────────────────────────────

    int imageSize = 3 * height * width;
    std::vector<float> imageData(imageSize);
    imageTensor.GetData(imageData.data(), imageSize * sizeof(float));

    // Save raw VAE output for comparison
    {
        auto dumpDir = fs::path(outputPath).parent_path();
        auto dumpImage = dumpDir / "debug_vae_output.bin";
        std::ofstream f(dumpImage, std::ios::binary);
        f.write((char*)imageData.data(), imageData.size() * sizeof(float));
        printf("  Saved raw VAE output: %s (%zu floats)\n", dumpImage.string().c_str(), imageData.size());
    }

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

    // Exit immediately to avoid Dawn cleanup crashes
    _exit(0);
}
