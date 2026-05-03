#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

// ─── GPT-2 byte-level BPE encoding ──────────────────────────────────────────
//
// GPT-2 maps each byte (0-255) to a printable Unicode character.
// This avoids control characters in the vocabulary.
// The mapping is:
//   '!' (33) .. '~' (126)  -> themselves
//   '\xA1' (161) .. '\xAC' (172) -> themselves
//   '\xAE' (174) .. '\xFF' (255) -> themselves
//   Remaining 0-32, 127-160, 173 -> 256..288 (Ā, ā, Ă, ă, ... Ġ for space)
//
// So 'Ġ' (U+0120 = 288) represents byte 0x20 (space).

static void build_byte_tables(
        std::unordered_map<uint32_t, uint8_t>& unicode_to_byte,
        uint32_t byte_to_unicode[256]) {
    // Build the GPT-2 byte encoder table
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_unicode[b] = (uint32_t)b;
        } else {
            byte_to_unicode[b] = 256 + n;
            n++;
        }
        unicode_to_byte[byte_to_unicode[b]] = (uint8_t)b;
    }
}

/// Decode a BPE token string (which uses GPT-2 byte encoding) to raw UTF-8 bytes.
static std::string bpe_token_to_bytes(
        const std::string& token,
        const std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
    std::string result;
    const uint8_t* p = (const uint8_t*)token.data();
    const uint8_t* end = p + token.size();
    while (p < end) {
        uint32_t cp = 0;
        int len = 0;
        // Decode UTF-8 codepoint
        if ((*p & 0x80) == 0)       { cp = *p; len = 1; }
        else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F; len = 2; }
        else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F; len = 3; }
        else if ((*p & 0xF8) == 0xF0) { cp = *p & 0x07; len = 4; }
        else { p++; continue; }
        for (int i = 1; i < len && p + i < end; i++)
            cp = (cp << 6) | (p[i] & 0x3F);
        p += len;

        auto it = unicode_to_byte.find(cp);
        if (it != unicode_to_byte.end()) {
            result.push_back((char)it->second);
        } else {
            // Not in byte mapping — pass through (e.g. special tokens)
            // Re-encode the codepoint as UTF-8
            if (cp < 0x80) result.push_back((char)cp);
            else if (cp < 0x800) {
                result.push_back((char)(0xC0 | (cp >> 6)));
                result.push_back((char)(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                result.push_back((char)(0xE0 | (cp >> 12)));
                result.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                result.push_back((char)(0x80 | (cp & 0x3F)));
            }
        }
    }
    return result;
}

/// Encode raw bytes to GPT-2 BPE token string
static std::string bytes_to_bpe_string(const std::string& text,
                                        const uint32_t byte_to_unicode[256]) {
    std::string result;
    for (uint8_t b : text) {
        uint32_t cp = byte_to_unicode[b];
        // Encode codepoint as UTF-8
        if (cp < 0x80) {
            result.push_back((char)cp);
        } else if (cp < 0x800) {
            result.push_back((char)(0xC0 | (cp >> 6)));
            result.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            result.push_back((char)(0xE0 | (cp >> 12)));
            result.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    return result;
}

// ─── Load from GGUF ──────────────────────────────────────────────────────────

bool Tokenizer::load(const GGUFFile& gguf) {
    // Extract vocabulary
    auto it = gguf.metadata.find("tokenizer.ggml.tokens");
    if (it == gguf.metadata.end()) {
        fprintf(stderr, "No tokenizer.ggml.tokens in GGUF\n");
        return false;
    }
    auto* tokens = std::get_if<std::vector<std::string>>(&it->second);
    if (!tokens) return false;

    vocab = *tokens;
    token_to_id.reserve(vocab.size());
    for (int32_t i = 0; i < (int32_t)vocab.size(); i++)
        token_to_id[vocab[i]] = i;

    // Extract merges
    auto mit = gguf.metadata.find("tokenizer.ggml.merges");
    if (mit != gguf.metadata.end()) {
        auto* merges = std::get_if<std::vector<std::string>>(&mit->second);
        if (merges) {
            for (int32_t i = 0; i < (int32_t)merges->size(); i++)
                merge_rank[(*merges)[i]] = i;
        }
    }

    // Special tokens
    eos_token_id = (int32_t)gguf.getU32("tokenizer.ggml.eos_token_id", 151645);
    bos_token_id = (int32_t)gguf.getU32("tokenizer.ggml.bos_token_id", 151643);

    // Build byte encoding tables
    uint32_t byte_to_unicode[256];
    build_byte_tables(unicode_to_byte_, byte_to_unicode);

    fprintf(stderr, "  Tokenizer: %zu tokens, %zu merges, EOS=%d\n",
           vocab.size(), merge_rank.size(), eos_token_id);

    return true;
}

// ─── Decode ──────────────────────────────────────────────────────────────────

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab.size())
        return "";
    const std::string& tok = vocab[token_id];
    // Special tokens (like <|im_end|>) — return as-is
    if (tok.size() >= 2 && tok[0] == '<' && tok.back() == '>')
        return tok;
    return bpe_token_to_bytes(tok, unicode_to_byte_);
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    for (auto id : token_ids)
        result += decode_token(id);
    return result;
}

// ─── Encode ──────────────────────────────────────────────────────────────────
//
// GPT-2 BPE encoding:
// 1. Convert input text bytes to BPE token strings
// 2. Split into individual characters (in BPE encoding)
// 3. Iteratively merge the highest-priority pair until no more merges

/// Split a BPE-encoded string into individual UTF-8 characters
static std::vector<std::string> split_to_chars(const std::string& s) {
    std::vector<std::string> chars;
    const uint8_t* p = (const uint8_t*)s.data();
    const uint8_t* end = p + s.size();
    while (p < end) {
        int len = 1;
        if ((*p & 0x80) == 0)       len = 1;
        else if ((*p & 0xE0) == 0xC0) len = 2;
        else if ((*p & 0xF0) == 0xE0) len = 3;
        else if ((*p & 0xF8) == 0xF0) len = 4;
        if (p + len > end) len = (int)(end - p);
        chars.emplace_back((const char*)p, len);
        p += len;
    }
    return chars;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    // 1. Convert bytes to BPE string representation
    uint32_t byte_to_unicode[256];
    {
        std::unordered_map<uint32_t, uint8_t> dummy;
        build_byte_tables(dummy, byte_to_unicode);
    }
    std::string bpe_text = bytes_to_bpe_string(text, byte_to_unicode);

    // 2. Split into characters (each is a BPE token candidate)
    auto pieces = split_to_chars(bpe_text);

    // 3. Iteratively merge the best pair
    while (pieces.size() > 1) {
        // Find the pair with the lowest merge rank
        int best_idx = -1;
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        for (int i = 0; i < (int)pieces.size() - 1; i++) {
            std::string pair = pieces[i] + " " + pieces[i + 1];
            auto it = merge_rank.find(pair);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;  // No more merges possible

        // Merge the pair
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }

    // 4. Convert merged pieces to token IDs
    std::vector<int32_t> ids;
    ids.reserve(pieces.size());
    for (auto& p : pieces) {
        auto it = token_to_id.find(p);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown token — encode each byte individually
            for (uint8_t b : p) {
                std::string single(1, (char)byte_to_unicode[b]);
                auto sit = token_to_id.find(single);
                if (sit != token_to_id.end())
                    ids.push_back(sit->second);
            }
        }
    }

    return ids;
}
