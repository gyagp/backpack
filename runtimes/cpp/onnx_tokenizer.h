#pragma once
/**
 * onnx_tokenizer.h — BPE tokenizer loaded from tokenizer.json (HuggingFace format).
 *
 * Supports GPT-2 style byte-level BPE (used by Phi, LLaMA, etc.)
 * Vocabulary and merges are extracted from tokenizer.json,
 * special tokens from config.json.
 */

#include <string>
#include <vector>
#include <unordered_map>

struct OnnxTokenizer {
    // Vocabulary: token_id -> token string (in BPE byte encoding)
    std::vector<std::string> vocab;

    // Reverse map: token string -> token_id
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges: ordered list of (piece_a, piece_b) pairs
    // merge_rank[pair_string] -> priority (lower = merge first)
    std::unordered_map<std::string, int32_t> merge_rank;

    // Added/special tokens: exact string -> token_id
    // These are matched before BPE encoding (e.g. <|im_start|>, <think>)
    std::unordered_map<std::string, int32_t> added_tokens;
    // Sorted by length (longest first) for greedy matching
    std::vector<std::string> added_tokens_sorted;

    // Special token IDs
    int32_t eos_token_id = -1;
    int32_t bos_token_id = -1;

    // --- API ---

    /// Load tokenizer from model directory (tokenizer.json + config.json)
    bool load(const std::string& modelDir);

    /// Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    /// Decode a single token ID to text
    std::string decode_token(int32_t token_id) const;

    /// Decode a sequence of token IDs to text
    std::string decode(const std::vector<int32_t>& token_ids) const;

private:
    // GPT-2 byte-level encoding tables
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
    uint32_t byte_to_unicode_[256]{};

    // BPE encode a text segment (no added token handling)
    std::vector<int32_t> encodeBpe(const std::string& text) const;
};
