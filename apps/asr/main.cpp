/**
 * Backpack ASR — Speech-to-text using bp::Session (Layer 1 API).
 *
 * Supports Qwen3-ASR pipeline:
 *   WAV → Mel Spectrogram → Encoder Conv → Encoder Transformer →
 *   Embedding Fusion → Decoder Init → Decoder Step (autoregressive) → Text
 *
 * All pipeline orchestration is in this app. The runtime (backpack.dll)
 * only provides Device, Model, Tensor, Session.
 *
 * Usage:
 *   backpack_asr --model path/to/qwen3-asr/ --audio speech.wav
 */

#include "backpack.h"
#include "gpu_context.h"
#include "onnx_tokenizer.h"
#include "../common/app_common.h"

#include "wav_reader.h"
#include "mel_spectrogram.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Special token IDs for Qwen3-ASR ────────────────────────────────────────

static constexpr int32_t IM_START_ID     = 151644;
static constexpr int32_t IM_END_ID       = 151645;
static constexpr int32_t ENDOFTEXT_ID    = 151643;
static constexpr int32_t AUDIO_START_ID  = 151669;
static constexpr int32_t AUDIO_END_ID    = 151670;
static constexpr int32_t AUDIO_PAD_ID    = 151676;
static constexpr int32_t NEWLINE_ID      = 198;

static constexpr int HIDDEN_SIZE = 1024;
static constexpr int VOCAB_SIZE  = 151936;

// ─── Helpers ────────────────────────────────────────────────────────────────

static int argmaxFloat(const float* data, int n) {
    int best = 0;
    float bestVal = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > bestVal) { bestVal = data[i]; best = i; }
    }
    return best;
}

static void printTensorStats(const char* label, const float* data, size_t n) {
    if (n == 0) { printf("  %s: empty\n", label); return; }
    float minV = 1e30f, maxV = -1e30f;
    double sumV = 0.0;
    for (size_t i = 0; i < n; i++) {
        minV = std::min(minV, data[i]);
        maxV = std::max(maxV, data[i]);
        sumV += data[i];
    }
    printf("  %s: min=%.4f max=%.4f avg=%.6f (%zu elements)\n",
           label, minV, maxV, sumV / n, n);
}

// ─── Load embed_tokens.bin ──────────────────────────────────────────────────

static bool loadEmbedTokens(const std::string& path, std::vector<float>& embeddings) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        return false;
    }
    auto fileSize = f.tellg();
    f.seekg(0);

    size_t expectedSize = (size_t)VOCAB_SIZE * HIDDEN_SIZE * sizeof(float);
    if ((size_t)fileSize != expectedSize) {
        fprintf(stderr, "Error: embed_tokens.bin size mismatch: got %lld, expected %zu\n",
                (long long)fileSize, expectedSize);
        return false;
    }

    embeddings.resize((size_t)VOCAB_SIZE * HIDDEN_SIZE);
    f.read(reinterpret_cast<char*>(embeddings.data()), expectedSize);
    printf("  Loaded embed_tokens.bin: [%d, %d] (%.0f MB)\n",
           VOCAB_SIZE, HIDDEN_SIZE, (float)expectedSize / (1024 * 1024));
    return true;
}

// ─── Build ASR prompt tokens ────────────────────────────────────────────────

static std::vector<int32_t> buildPromptTokens(int numAudioTokens,
                                               const std::string& language) {
    // Format:
    // <|im_start|>system\n<|im_end|>\n
    // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    // <|im_start|>assistant\n

    std::vector<int32_t> tokens;

    // System turn (empty)
    tokens.push_back(IM_START_ID);
    // "system" in BPE — for simplicity, we'll encode "system" properly
    // but since we have the tokenizer we can use it for the text parts
    // For now, hardcode the known token sequence:
    // system = [9125]  \n = [198]
    tokens.push_back(9125);   // "system"
    tokens.push_back(NEWLINE_ID);
    tokens.push_back(IM_END_ID);
    tokens.push_back(NEWLINE_ID);

    // User turn with audio
    tokens.push_back(IM_START_ID);
    tokens.push_back(872);    // "user"
    tokens.push_back(NEWLINE_ID);
    tokens.push_back(AUDIO_START_ID);
    for (int i = 0; i < numAudioTokens; i++) {
        tokens.push_back(AUDIO_PAD_ID);
    }
    tokens.push_back(AUDIO_END_ID);
    tokens.push_back(IM_END_ID);
    tokens.push_back(NEWLINE_ID);

    // Assistant turn
    tokens.push_back(IM_START_ID);
    tokens.push_back(77091);  // "assistant"
    tokens.push_back(NEWLINE_ID);

    return tokens;
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string modelDir, audioPath, backendStr;
    std::string language;
    int maxTokens = 512;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)       modelDir = argv[++i];
        else if (arg == "--audio" && i+1 < argc)   audioPath = argv[++i];
        else if (arg == "--language" && i+1 < argc) language = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) maxTokens = atoi(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) backendStr = argv[++i];
    }

    if (modelDir.empty() || audioPath.empty()) {
        fprintf(stderr,
            "Backpack ASR — WebGPU speech-to-text\n\n"
            "Usage: %s --model <dir> --audio <wav> [options]\n\n"
            "  --language <lang>   Language hint (default: auto-detect)\n"
            "  --max-tokens <n>    Max output tokens (default: 512)\n"
            "  --backend <name>    vulkan / d3d12 / metal\n\n"
            "Model directory should contain:\n"
            "  onnx_models/encoder_conv.onnx\n"
            "  onnx_models/encoder_transformer.onnx\n"
            "  onnx_models/decoder_init.onnx  (fp32)\n"
            "  onnx_models/decoder_step.onnx  (fp32)\n"
            "  embed_tokens.bin\n"
            "  tokenizer.json + config.json\n", argv[0]);
        return 1;
    }

    printf("Backpack ASR: Qwen3-ASR-0.6B\n");
    printf("Audio: %s\n", audioPath.c_str());
    if (!language.empty()) printf("Language: %s\n", language.c_str());

    auto tTotal = std::chrono::steady_clock::now();

    // ─── 1. Read WAV file ────────────────────────────────────────────────

    printf("\n--- Audio Input ---\n");
    std::vector<float> audioSamples;
    if (!asr::readWav(audioPath, audioSamples)) return 1;

    // ─── 2. Compute mel spectrogram ──────────────────────────────────────

    printf("\n--- Mel Spectrogram ---\n");
    auto tMel0 = std::chrono::steady_clock::now();
    int nFrames = 0;
    auto melSpec = asr::computeLogMelSpectrogram(audioSamples, nFrames);
    auto tMel1 = std::chrono::steady_clock::now();
    auto melMs = std::chrono::duration<double, std::milli>(tMel1 - tMel0).count();
    printf("  Mel spectrogram: %.0fms\n", melMs);
    printTensorStats("Mel", melSpec.data(), melSpec.size());

    // ─── 3. Create device ────────────────────────────────────────────────

    auto device = app::createDevice(backendStr);
    if (!device.IsValid()) { fprintf(stderr, "GPU init failed\n"); return 1; }
    app::printGpuInfo(device);
    printf("\n");

    // ─── 4. Load tokenizer ───────────────────────────────────────────────

    OnnxTokenizer tokenizer;
    if (!tokenizer.load(modelDir)) {
        fprintf(stderr, "Error: failed to load tokenizer from %s\n", modelDir.c_str());
        return 1;
    }
    printf("Tokenizer: %zu vocab entries\n", tokenizer.vocab.size());

    // ─── 5. Load embed_tokens.bin ────────────────────────────────────────

    std::vector<float> embedTokens;
    auto embedPath = (fs::path(modelDir) / "embed_tokens.bin").string();
    if (!loadEmbedTokens(embedPath, embedTokens)) return 1;

    // ─── 6. Encoder Conv ─────────────────────────────────────────────────

    printf("\n--- Encoder Conv ---\n"); fflush(stdout);
    auto tEnc0 = std::chrono::steady_clock::now();

    // Resolve model paths (check onnx_models/ subdir first, then root)
    auto resolveModelPath = [&](const std::string& name) -> std::string {
        auto subdir = (fs::path(modelDir) / "onnx_models" / name).string();
        if (fs::exists(subdir)) return subdir;
        return (fs::path(modelDir) / name).string();
    };

    // Chunk the mel spectrogram for the encoder conv
    // The reference implementation processes in chunks of CHUNK_SIZE frames
    constexpr int CHUNK_SIZE = 100;
    int nChunks = (nFrames + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int maxChunkLen = CHUNK_SIZE;

    // Pad mel data into [nChunks, 1, N_MELS, maxChunkLen] tensor
    // mel input format: melSpec is [N_MELS, nFrames] row-major
    std::vector<float> melChunked(nChunks * 1 * asr::N_MELS * maxChunkLen, 0.0f);
    for (int c = 0; c < nChunks; c++) {
        int startFrame = c * CHUNK_SIZE;
        int endFrame = std::min(startFrame + CHUNK_SIZE, nFrames);
        int chunkFrames = endFrame - startFrame;
        for (int m = 0; m < asr::N_MELS; m++) {
            for (int t = 0; t < chunkFrames; t++) {
                melChunked[c * asr::N_MELS * maxChunkLen + m * maxChunkLen + t] =
                    melSpec[m * nFrames + startFrame + t];
            }
        }
    }

    auto encoderConvPath = resolveModelPath("encoder_conv.onnx");
    printf("Loading encoder_conv: %s\n", encoderConvPath.c_str()); fflush(stdout);
    auto encoderConv = bp::Model::Load(device, encoderConvPath);

    auto melInput = bp::Tensor::Create(device, bp::DataType::Float32,
        {(int64_t)nChunks, 1, (int64_t)asr::N_MELS, (int64_t)maxChunkLen});
    melInput.SetData(melChunked.data(), melChunked.size() * sizeof(float));

    // Create output tensor (shape will be determined by the model)
    auto convOutput = bp::Tensor::Create(device, bp::DataType::Float32, {1});

    auto convSession = bp::Session::Create(encoderConv);
    convSession.SetInput("mel", melInput);
    convSession.SetOutput("conv_features", convOutput);
    convSession.Run();

    // Read conv output shape and data
    auto convShape = convOutput.GetShape();
    int64_t convElements = 1;
    for (auto d : convShape) convElements *= d;
    std::vector<float> convData((size_t)convElements);
    convOutput.GetData(convData.data(), convData.size() * sizeof(float));
    printf("  Conv output shape: [");
    for (size_t i = 0; i < convShape.size(); i++)
        printf("%s%lld", i ? "," : "", (long long)convShape[i]);
    printf("]\n");
    printTensorStats("Conv output", convData.data(), convData.size());

    auto tEnc1 = std::chrono::steady_clock::now();
    auto encConvMs = std::chrono::duration<double, std::milli>(tEnc1 - tEnc0).count();
    printf("  Encoder conv: %.0fms\n", encConvMs);

    convSession.Release();
    encoderConv.Release();

    // Flush GPU memory before loading next model
    {
        auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
        gpuCtx->waitForQueue();
        gpuCtx->flushBufferPool();
    }

    // ─── 7. Encoder Transformer ──────────────────────────────────────────

    printf("\n--- Encoder Transformer ---\n"); fflush(stdout);
    auto tEncT0 = std::chrono::steady_clock::now();

    auto encoderTransPath = resolveModelPath("encoder_transformer.onnx");
    printf("Loading encoder_transformer: %s\n", encoderTransPath.c_str()); fflush(stdout);
    auto encoderTrans = bp::Model::Load(device, encoderTransPath);

    // The encoder transformer takes the conv output and produces audio features
    auto audioFeatures = bp::Tensor::Create(device, bp::DataType::Float32, {1});

    auto transSession = bp::Session::Create(encoderTrans);
    transSession.SetInput("hidden_states", convOutput);
    transSession.SetOutput("audio_features", audioFeatures);
    transSession.Run();

    auto audioShape = audioFeatures.GetShape();
    int64_t audioElements = 1;
    for (auto d : audioShape) audioElements *= d;
    std::vector<float> audioData((size_t)audioElements);
    audioFeatures.GetData(audioData.data(), audioData.size() * sizeof(float));
    printf("  Audio features shape: [");
    for (size_t i = 0; i < audioShape.size(); i++)
        printf("%s%lld", i ? "," : "", (long long)audioShape[i]);
    printf("]\n");
    printTensorStats("Audio features", audioData.data(), audioData.size());

    // Determine number of audio tokens from encoder output
    // Audio features shape is typically [1, numAudioTokens, HIDDEN_SIZE]
    int numAudioTokens = 1;
    if (audioShape.size() >= 2) numAudioTokens = (int)audioShape[audioShape.size() - 2];
    printf("  Audio tokens: %d\n", numAudioTokens);

    auto tEncT1 = std::chrono::steady_clock::now();
    auto encTransMs = std::chrono::duration<double, std::milli>(tEncT1 - tEncT0).count();
    printf("  Encoder transformer: %.0fms\n", encTransMs);

    transSession.Release();
    encoderTrans.Release();
    melInput.Release();

    {
        auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
        gpuCtx->waitForQueue();
        gpuCtx->flushBufferPool();
    }

    // ─── 8. Build prompt + fuse embeddings ───────────────────────────────

    printf("\n--- Embedding Fusion ---\n");
    auto tEmb0 = std::chrono::steady_clock::now();

    auto promptTokens = buildPromptTokens(numAudioTokens, language);
    int seqLen = (int)promptTokens.size();
    printf("  Prompt tokens: %d (including %d audio pads)\n", seqLen, numAudioTokens);

    // Build input_embeds by looking up embeddings and fusing audio features
    std::vector<float> inputEmbeds(seqLen * HIDDEN_SIZE, 0.0f);
    int audioIdx = 0;
    for (int i = 0; i < seqLen; i++) {
        int32_t tid = promptTokens[i];
        if (tid == AUDIO_PAD_ID && audioIdx < numAudioTokens) {
            // Replace with audio feature embedding
            memcpy(&inputEmbeds[i * HIDDEN_SIZE],
                   &audioData[audioIdx * HIDDEN_SIZE],
                   HIDDEN_SIZE * sizeof(float));
            audioIdx++;
        } else {
            // Look up from embed_tokens
            if (tid >= 0 && tid < VOCAB_SIZE) {
                memcpy(&inputEmbeds[i * HIDDEN_SIZE],
                       &embedTokens[(size_t)tid * HIDDEN_SIZE],
                       HIDDEN_SIZE * sizeof(float));
            }
        }
    }

    auto tEmb1 = std::chrono::steady_clock::now();
    auto embMs = std::chrono::duration<double, std::milli>(tEmb1 - tEmb0).count();
    printf("  Embedding fusion: %.0fms\n", embMs);
    printTensorStats("Input embeds", inputEmbeds.data(), inputEmbeds.size());

    // ─── 9. Decoder Init (Prefill) ───────────────────────────────────────

    printf("\n--- Decoder Init ---\n"); fflush(stdout);
    auto tDec0 = std::chrono::steady_clock::now();

    auto decoderInitPath = resolveModelPath("decoder_init.onnx");
    if (!fs::exists(decoderInitPath)) {
        // Fallback to int8 variant if fp32 not available
        decoderInitPath = resolveModelPath("decoder_init.int8.onnx");
    }
    printf("Loading decoder_init: %s\n", decoderInitPath.c_str()); fflush(stdout);
    auto decoderInit = bp::Model::Load(device, decoderInitPath);

    // Print model I/O info
    printf("  Inputs: %d, Outputs: %d\n",
           decoderInit.GetInputCount(), decoderInit.GetOutputCount());
    for (int i = 0; i < decoderInit.GetInputCount(); i++) {
        auto info = decoderInit.GetInputInfo(i);
        printf("    Input[%d]: %s shape=[", i, info.name.c_str());
        for (size_t j = 0; j < info.shape.size(); j++)
            printf("%s%lld", j ? "," : "", (long long)info.shape[j]);
        printf("]\n");
    }
    for (int i = 0; i < decoderInit.GetOutputCount(); i++) {
        auto info = decoderInit.GetOutputInfo(i);
        printf("    Output[%d]: %s shape=[", i, info.name.c_str());
        for (size_t j = 0; j < info.shape.size(); j++)
            printf("%s%lld", j ? "," : "", (long long)info.shape[j]);
        printf("]\n");
    }

    // Create input tensors
    auto inputEmbedsTensor = bp::Tensor::Create(device, bp::DataType::Float32,
        {1, (int64_t)seqLen, (int64_t)HIDDEN_SIZE});
    inputEmbedsTensor.SetData(inputEmbeds.data(), inputEmbeds.size() * sizeof(float));

    std::vector<int64_t> posIds(seqLen);
    for (int i = 0; i < seqLen; i++) posIds[i] = i;
    auto posIdsTensor = bp::Tensor::Create(device, bp::DataType::Int64,
        {1, (int64_t)seqLen});
    posIdsTensor.SetData(posIds.data(), seqLen * sizeof(int64_t));

    // Output tensors — logits + KV cache
    auto logitsTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
    auto presentKeysTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});
    auto presentValuesTensor = bp::Tensor::Create(device, bp::DataType::Float32, {1});

    auto initSession = bp::Session::Create(decoderInit);
    initSession.SetInput("input_embeds", inputEmbedsTensor);
    initSession.SetInput("position_ids", posIdsTensor);

    // Set outputs — use model's declared output names
    for (int i = 0; i < decoderInit.GetOutputCount(); i++) {
        auto name = decoderInit.GetOutputName(i);
        if (name == "logits") {
            initSession.SetOutput(name, logitsTensor);
        } else if (name.find("key") != std::string::npos ||
                   name.find("present") != std::string::npos) {
            if (name.find("key") != std::string::npos) {
                initSession.SetOutput(name, presentKeysTensor);
            }
        }
        // For MVP, let other outputs auto-allocate
    }
    // Simpler approach: just set logits output, let everything else auto-allocate
    initSession.Reset();
    initSession.SetInput("input_embeds", inputEmbedsTensor);
    initSession.SetInput("position_ids", posIdsTensor);
    initSession.SetOutput("logits", logitsTensor);

    printf("  Running decoder_init (seq_len=%d)...\n", seqLen); fflush(stdout);
    initSession.Run();

    // Read logits and argmax
    auto logitsShape = logitsTensor.GetShape();
    printf("  Logits shape: [");
    for (size_t i = 0; i < logitsShape.size(); i++)
        printf("%s%lld", i ? "," : "", (long long)logitsShape[i]);
    printf("]\n");

    int64_t logitsElements = 1;
    for (auto d : logitsShape) logitsElements *= d;
    std::vector<float> logitsData((size_t)logitsElements);
    logitsTensor.GetData(logitsData.data(), logitsData.size() * sizeof(float));

    // Argmax the last position's logits
    int lastPosOffset = (seqLen - 1) * VOCAB_SIZE;
    int firstToken = argmaxFloat(&logitsData[lastPosOffset], VOCAB_SIZE);
    printf("  First token: %d (%s)\n", firstToken,
           tokenizer.decode_token(firstToken).c_str());

    auto tDec1 = std::chrono::steady_clock::now();
    auto decInitMs = std::chrono::duration<double, std::milli>(tDec1 - tDec0).count();
    printf("  Decoder init: %.0fms\n", decInitMs);

    initSession.Release();
    decoderInit.Release();
    inputEmbedsTensor.Release();
    posIdsTensor.Release();

    {
        auto* gpuCtx = static_cast<GPUContext*>(device.GetGPUContext());
        gpuCtx->waitForQueue();
        gpuCtx->flushBufferPool();
    }

    // ─── 10. Autoregressive Decode Loop ──────────────────────────────────

    printf("\n--- Decoding ---\n");
    printf("Output: "); fflush(stdout);

    auto tDecLoop0 = std::chrono::steady_clock::now();

    // For the full decoder step loop, we need the KV cache from decoder_init.
    // For now, we use a simplified approach: just report the first token
    // and note that full decode requires decoder_step.onnx with KV cache passing.

    std::string outputText;
    std::vector<int32_t> outputTokens;
    outputTokens.push_back(firstToken);

    auto decoderStepPath = resolveModelPath("decoder_step.onnx");
    bool hasDecoderStep = fs::exists(decoderStepPath);
    if (!hasDecoderStep) {
        decoderStepPath = resolveModelPath("decoder_step.int8.onnx");
        hasDecoderStep = fs::exists(decoderStepPath);
    }

    if (hasDecoderStep) {
        printf("Loading decoder_step: %s\n", decoderStepPath.c_str()); fflush(stdout);
        auto decoderStep = bp::Model::Load(device, decoderStepPath);

        // Print model I/O info for decoder_step
        printf("  Step inputs: %d, Step outputs: %d\n",
               decoderStep.GetInputCount(), decoderStep.GetOutputCount());
        for (int i = 0; i < decoderStep.GetInputCount(); i++) {
            auto info = decoderStep.GetInputInfo(i);
            printf("    Input[%d]: %s\n", i, info.name.c_str());
        }
        for (int i = 0; i < decoderStep.GetOutputCount(); i++) {
            auto info = decoderStep.GetOutputInfo(i);
            printf("    Output[%d]: %s\n", i, info.name.c_str());
        }

        // TODO: Implement full decode loop once we understand the exact
        // KV cache tensor names and shapes from the model I/O inspection.
        // The pattern is:
        //   1. Look up token embedding from embedTokens
        //   2. Set input_embeds [1, 1, 1024], position_ids [1, 1]
        //   3. Set past_key_values from previous step
        //   4. Run → get logits + present_key_values
        //   5. Argmax → next token
        //   6. Check for EOS (IM_END_ID or ENDOFTEXT_ID)
        //   7. Repeat

        printf("\n  (Full decode loop implementation pending — "
               "need to match KV cache tensor names)\n");

        decoderStep.Release();
    }

    // Print first token output
    std::string firstTokenText = tokenizer.decode_token(firstToken);
    printf("%s", firstTokenText.c_str());

    printf("\n");

    auto tDecLoop1 = std::chrono::steady_clock::now();
    auto decLoopMs = std::chrono::duration<double, std::milli>(tDecLoop1 - tDecLoop0).count();

    // ─── Summary ─────────────────────────────────────────────────────────

    auto tEnd = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration<double, std::milli>(tEnd - tTotal).count();
    printf("\n--- Performance ---\n");
    printf("  Total:       %.0fms\n", totalMs);
    printf("  Mel:         %.0fms\n", melMs);
    printf("  Encoder conv:  %.0fms\n", encConvMs);
    printf("  Encoder trans: %.0fms\n", encTransMs);
    printf("  Embed fusion:  %.0fms\n", embMs);
    printf("  Decoder init:  %.0fms\n", decInitMs);
    printf("  Decode loop:   %.0fms (%d tokens)\n", decLoopMs,
           (int)outputTokens.size());

    // Exit immediately to avoid Dawn cleanup crashes
    _exit(0);
}
