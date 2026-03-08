#pragma once
/**
 * tokenizer.h -- BPE tokenizer loaded from GGUF metadata.
 *
 * Supports GPT-2 style byte-level BPE (used by Qwen, LLaMA-3, etc.)
 * Vocabulary, merges, and special tokens are all extracted from GGUF.
 */

#include "gguf_loader.h"
#include <string>
#include <vector>
#include <unordered_map>

struct Tokenizer {
    // Vocabulary: token_id -> token string (in BPE byte encoding)
    std::vector<std::string> vocab;

    // Reverse map: token string -> token_id
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges: ordered list of (piece_a, piece_b) pairs
    // merge_rank[pair_string] -> priority (lower = merge first)
    std::unordered_map<std::string, int32_t> merge_rank;

    // Special token IDs
    int32_t eos_token_id = -1;
    int32_t bos_token_id = -1;

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
};
