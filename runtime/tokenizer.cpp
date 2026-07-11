#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>

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
    // Detect tokenizer model.
    std::string model_str = gguf.getString("tokenizer.ggml.model", "gpt2");
    if (model_str == "gpt2")        model_kind = Model::Gpt2Bpe;
    else if (model_str == "llama")  model_kind = Model::LlamaSpm;
    // Gemma's SentencePiece vocab is declared under model-specific strings in
    // newer GGUFs ("gemma", "gemma2", "gemma3", "gemma4", ...). They are all
    // llama-style SPM (scores + greedy merge, ▁ word-start), so treat any
    // "gemma*" as SPM rather than falling back to the wrong GPT-2 BPE path.
    else if (model_str.rfind("gemma", 0) == 0) model_kind = Model::LlamaSpm;
    else                            model_kind = Model::Unknown;

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

    // Extract merges (GPT-2 BPE only).
    auto mit = gguf.metadata.find("tokenizer.ggml.merges");
    if (mit != gguf.metadata.end()) {
        auto* merges = std::get_if<std::vector<std::string>>(&mit->second);
        if (merges) {
            for (int32_t i = 0; i < (int32_t)merges->size(); i++)
                merge_rank[(*merges)[i]] = i;
        }
    }

    // Extract scores (SPM uses these to pick merges).
    // GGUF stores scores as a float32 array — teach the loader (and our
    // variant) to keep that array so we can read it here.
    {
        auto sit = gguf.metadata.find("tokenizer.ggml.scores");
        if (sit != gguf.metadata.end()) {
            if (auto* sf = std::get_if<std::vector<float>>(&sit->second)) {
                scores = *sf;
            }
        }
    }

    // Special tokens
    eos_token_id = (int32_t)gguf.getU32("tokenizer.ggml.eos_token_id", 151645);
    bos_token_id = (int32_t)gguf.getU32("tokenizer.ggml.bos_token_id", -1);
    add_bos_token = gguf.getBool("tokenizer.ggml.add_bos_token",
                                 model_kind == Model::LlamaSpm);
    add_space_prefix = gguf.getBool("tokenizer.ggml.add_space_prefix",
                                    model_kind == Model::LlamaSpm);

    // Build byte encoding tables (used by GPT-2 BPE and by decode of GPT-2
    // models; SPM tokens decode to raw bytes directly).
    uint32_t byte_to_unicode[256];
    build_byte_tables(unicode_to_byte_, byte_to_unicode);

    // Build list of special tokens (control=3 and user_defined=4) for
    // pre-BPE matching. Without this, multi-byte specials like "<|im_start|>"
    // tokenize as raw bytes instead of their single dedicated ID.
    // Also build the byte-fallback table (token_type=6) for SPM models.
    byte_to_token.assign(256, -1);
    {
        auto tit = gguf.metadata.find("tokenizer.ggml.token_type");
        if (tit != gguf.metadata.end()) {
            auto* types = std::get_if<std::vector<int32_t>>(&tit->second);
            if (types) {
                for (int32_t i = 0; i < (int32_t)vocab.size() && i < (int32_t)types->size(); i++) {
                    int32_t ty = (*types)[i];
                    if (ty == 3 || ty == 4) {
                        // For GPT-2 vocabs the entry is in BPE byte encoding —
                        // decode back to raw bytes before string-matching.
                        // For SPM vocabs the entry is already raw UTF-8.
                        std::string raw = (model_kind == Model::Gpt2Bpe)
                            ? bpe_token_to_bytes(vocab[i], unicode_to_byte_)
                            : vocab[i];
                        if (!raw.empty())
                            special_tokens_sorted.emplace_back(std::move(raw), i);
                    } else if (ty == 6) {
                        // Byte-fallback token, vocab entry is "<0xHH>"
                        const std::string& s = vocab[i];
                        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x') {
                            auto hex = [](char c) -> int {
                                if (c >= '0' && c <= '9') return c - '0';
                                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                                return -1;
                            };
                            int hi = hex(s[3]), lo = hex(s[4]);
                            if (hi >= 0 && lo >= 0)
                                byte_to_token[(hi << 4) | lo] = i;
                        }
                    }
                }
                // Longest first so greedy match prefers e.g. "<|im_start|>" over "<".
                std::sort(special_tokens_sorted.begin(), special_tokens_sorted.end(),
                          [](const auto& a, const auto& b) {
                              return a.first.size() > b.first.size();
                          });
            }
        }
    }

    fprintf(stderr, "  Tokenizer: %zu tokens, %zu merges, EOS=%d, special=%zu, model=%s%s\n",
           vocab.size(), merge_rank.size(), eos_token_id, special_tokens_sorted.size(),
           model_kind == Model::LlamaSpm ? "spm" :
           model_kind == Model::Gpt2Bpe ? "gpt2" : "unknown",
           add_bos_token ? ", +bos" : "");

    return true;
}

// ─── Decode ──────────────────────────────────────────────────────────────────

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab.size())
        return "";
    const std::string& tok = vocab[token_id];
    if (model_kind == Model::LlamaSpm) {
        // SPM vocab is raw UTF-8. ▁ (U+2581) marks word starts and decodes
        // back to a space. Byte-fallback tokens (<0xHH>) decode to that byte.
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x'
            && tok[5] == '>') {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            int hi = hex(tok[3]), lo = hex(tok[4]);
            if (hi >= 0 && lo >= 0) {
                std::string out;
                out.push_back((char)((hi << 4) | lo));
                return out;
            }
        }
        // Replace ▁ (e2 96 81 in UTF-8) with space.
        std::string out;
        out.reserve(tok.size());
        for (size_t i = 0; i < tok.size(); ) {
            if (i + 2 < tok.size() &&
                (uint8_t)tok[i] == 0xE2 && (uint8_t)tok[i+1] == 0x96 &&
                (uint8_t)tok[i+2] == 0x81) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(tok[i]);
                i++;
            }
        }
        return out;
    }
    // GPT-2 BPE: special tokens like <|im_end|> decode as-is.
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

    // SPM adds the ▁ word-start prefix only at the very beginning of the input,
    // not after every special token. Track whether we are still at the start.
    bool atStart = true;
    auto encode_segment = [&](const std::string& s) {
        auto r = (model_kind == Model::LlamaSpm)
            ? encode_spm_segment(s, atStart)
            : encode_bpe_segment(s);
        atStart = false;
        return r;
    };

    // 0. Split text on special tokens (control + user_defined). Each special
    //    token must be matched as an exact string and emitted as its single
    //    dedicated ID — otherwise multi-byte specials like "<|im_start|>"
    //    BPE-tokenize as raw bytes and the chat template falls apart.
    if (!special_tokens_sorted.empty()) {
        std::vector<int32_t> ids;
        std::string buf;
        size_t pos = 0;
        while (pos < text.size()) {
            bool matched = false;
            for (auto& sp : special_tokens_sorted) {
                if (pos + sp.first.size() <= text.size() &&
                    text.compare(pos, sp.first.size(), sp.first) == 0) {
                    if (!buf.empty()) {
                        auto sub = encode_segment(buf);
                        ids.insert(ids.end(), sub.begin(), sub.end());
                        buf.clear();
                    }
                    ids.push_back(sp.second);
                    atStart = false;  // a special token consumed the start
                    pos += sp.first.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                buf.push_back(text[pos]);
                pos++;
            }
        }
        if (!buf.empty()) {
            auto sub = encode_segment(buf);
            ids.insert(ids.end(), sub.begin(), sub.end());
        }
        return ids;
    }

    return encode_segment(text);
}

std::vector<int32_t> Tokenizer::encode_bpe_segment(const std::string& text) const {
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

// ─── llama-style SentencePiece BPE encoder ──────────────────────────────────
//
// Algorithm (per llama.cpp's llm_tokenizer_spm):
//   1. Normalize input: prepend ▁ if add_space_prefix, replace each ' ' with ▁.
//   2. Split into UTF-8 codepoints; map each to its vocab id (or byte-fallback).
//   3. Greedy merge: repeatedly find the adjacent pair with the highest score
//      that exists in the vocab, merge it. Stop when no mergeable pair remains.
//   4. Emit ids in order.

std::vector<int32_t> Tokenizer::encode_spm_segment(const std::string& text, bool addPrefix) const {
    if (text.empty()) return {};

    // 1. Normalize: turn ASCII spaces into ▁ (U+2581). The leading word-start ▁
    // is only added when addPrefix (i.e. this is the very start of the input),
    // not for segments that follow a special token.
    std::string norm;
    norm.reserve(text.size() + 6);
    auto append_under = [&]() { norm.push_back((char)0xE2); norm.push_back((char)0x96); norm.push_back((char)0x81); };
    bool start = true;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (start && add_space_prefix && addPrefix && c != ' ') {
            append_under();
            start = false;
        } else if (c == ' ') {
            append_under();
            start = false;
            continue;
        } else {
            start = false;
        }
        norm.push_back(c);
    }
    if (norm.empty()) norm = text;

    // 2. Split into UTF-8 characters.
    auto chars = split_to_chars(norm);
    if (chars.empty()) return {};

    struct Sym { std::string s; int prev; int next; bool dead; };
    std::vector<Sym> syms;
    syms.reserve(chars.size());
    for (size_t i = 0; i < chars.size(); i++) {
        syms.push_back({chars[i], (int)i - 1,
                        (i + 1 < chars.size()) ? (int)i + 1 : -1, false});
    }

    // 3. Greedy merges by SPM score. Priority queue keyed by (score, length).
    struct Bigram { float score; int left; int right; size_t total_len; };
    auto cmp = [](const Bigram& a, const Bigram& b) {
        if (a.score != b.score) return a.score < b.score;
        return a.left > b.left;  // earlier left wins on tie
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> queue(cmp);

    auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        std::string merged = syms[left].s + syms[right].s;
        auto it = token_to_id.find(merged);
        if (it == token_to_id.end()) return;
        int32_t id = it->second;
        if (id < 0 || (size_t)id >= scores.size()) return;
        queue.push({scores[id], left, right, merged.size()});
    };

    for (size_t i = 0; i + 1 < syms.size(); i++) {
        try_add((int)i, (int)i + 1);
    }

    while (!queue.empty()) {
        Bigram top = queue.top(); queue.pop();
        Sym& L = syms[top.left];
        Sym& R = syms[top.right];
        if (L.dead || R.dead) continue;
        // Make sure the symbols still concatenate to the merged string we
        // queued — otherwise stale entry.
        if (L.s.size() + R.s.size() != top.total_len) continue;
        L.s += R.s;
        L.next = R.next;
        if (R.next >= 0) syms[R.next].prev = top.left;
        R.dead = true;
        try_add(L.prev, top.left);
        try_add(top.left, L.next);
    }

    // 4. Emit ids in order. Byte-fallback for any leftover symbol that didn't
    //    merge into a known token.
    std::vector<int32_t> ids;
    int idx = 0;
    while (idx >= 0) {
        const Sym& s = syms[idx];
        auto it = token_to_id.find(s.s);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            for (unsigned char b : s.s) {
                int32_t tid = byte_to_token[b];
                if (tid >= 0) ids.push_back(tid);
            }
        }
        idx = s.next;
    }
    return ids;
}
