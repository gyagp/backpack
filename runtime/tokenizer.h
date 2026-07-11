#pragma once
/**
 * tokenizer.h -- BPE tokenizer loaded from GGUF metadata.
 *
 * Supports GPT-2 style byte-level BPE (used by Qwen, LLaMA-3, etc.)
 * and llama-style SentencePiece BPE (used by Gemma, Mistral, Llama-1/2).
 * Vocabulary, merges, and special tokens are all extracted from GGUF.
 */

#include "gguf_loader.h"
#include <string>
#include <vector>
#include <unordered_map>

struct Tokenizer {
    enum class Model {
        Gpt2Bpe,     // tokenizer.ggml.model == "gpt2" (Qwen, LLaMA-3)
        LlamaSpm,    // tokenizer.ggml.model == "llama" (Gemma, Mistral, Llama-1/2)
        Unknown,
    };
    Model model_kind = Model::Gpt2Bpe;

    // Vocabulary: token_id -> token string. For GPT-2 BPE this is in GPT-2 byte
    // encoding (e.g. "Ġhello"); for SPM it's raw UTF-8 with ▁ for word starts.
    std::vector<std::string> vocab;

    // Reverse map: token string -> token_id
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges: ordered list of (piece_a, piece_b) pairs (GPT-2 BPE).
    // merge_rank[pair_string] -> priority (lower = merge first)
    std::unordered_map<std::string, int32_t> merge_rank;

    // SPM scores: per-vocab log-likelihood used to choose the optimal pair to
    // merge in llama-style BPE. Higher score = prefer this token. Length equals
    // vocab.size() when populated.
    std::vector<float> scores;

    // Byte-fallback table: byte (0..255) -> token id, or -1 if no byte token
    // for that byte. Populated for SPM models that include <0x00>..<0xFF>.
    std::vector<int32_t> byte_to_token;

    // Special tokens (control + user-defined). Encoded in raw UTF-8 — must be
    // matched on the input text BEFORE BPE so they tokenize as single IDs.
    // Sorted longest first for greedy matching.
    std::vector<std::pair<std::string, int32_t>> special_tokens_sorted;

    // Special token IDs
    int32_t eos_token_id = -1;
    int32_t bos_token_id = -1;
    bool add_bos_token = false;
    bool add_space_prefix = true;  // SPM only: prefix input with ▁

    // --- API ---

    /// Load tokenizer from GGUF metadata
    bool load(const GGUFFile& gguf);

    /// Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    /// Decode a single token ID to text
    std::string decode_token(int32_t token_id) const;

    /// Decode a sequence of token IDs to text
    std::string decode(const std::vector<int32_t>& token_ids) const;

private:
    // GPT-2 byte-level encoding tables
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

    // GPT-2 BPE-encode a non-special segment of text.
    std::vector<int32_t> encode_bpe_segment(const std::string& text) const;

    // Llama-SPM encode a non-special segment of text. Applies ▁ normalization
    // then greedy merge by score.
    std::vector<int32_t> encode_spm_segment(const std::string& text, bool addPrefix) const;
};
