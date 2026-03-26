#include "onnx_tokenizer.h"
#include "json_parser.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;

// ─── GPT-2 byte-level BPE encoding ──────────────────────────────────────────
// Same encoding as tokenizer.cpp — maps each byte to a printable Unicode char.

static void build_byte_tables(
        std::unordered_map<uint32_t, uint8_t>& unicode_to_byte,
        uint32_t byte_to_unicode[256]) {
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

/// Decode a BPE token string (GPT-2 byte encoding) to raw UTF-8 bytes.
static std::string bpe_token_to_bytes(
        const std::string& token,
        const std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
    std::string result;
    const uint8_t* p = (const uint8_t*)token.data();
    const uint8_t* end = p + token.size();
    while (p < end) {
        uint32_t cp = 0;
        int len = 0;
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

/// Encode raw bytes to GPT-2 BPE string representation.
static std::string bytes_to_bpe_string(const std::string& text,
                                        const uint32_t byte_to_unicode[256]) {
    std::string result;
    for (uint8_t b : text) {
        uint32_t cp = byte_to_unicode[b];
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

/// Split a UTF-8 string into individual characters.
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

// ─── Fast tokenizer.json parser ──────────────────────────────────────────────
// The generic json_parse is too slow for 15MB+ tokenizer files with 200K+ vocab.
// This specialized parser extracts only "model.vocab" and "model.merges".

namespace {

// Skip whitespace
static const char* skipWs(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    return p;
}

// Parse a JSON string, handling escape sequences. Returns pointer past closing quote.
static const char* parseJsonString(const char* p, const char* end, std::string& out) {
    out.clear();
    if (p >= end || *p != '"') return p;
    p++; // skip opening quote
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            p++;
            switch (*p) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '/': out += '/'; break;
                case 'u': {
                    // \uXXXX — parse 4 hex digits
                    if (p + 4 < end) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        uint32_t cp = (uint32_t)strtoul(hex, nullptr, 16);
                        p += 4;
                        // Encode as UTF-8
                        if (cp < 0x80) {
                            out += (char)cp;
                        } else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: out += *p; break;
            }
        } else {
            out += *p;
        }
        p++;
    }
    if (p < end) p++; // skip closing quote
    return p;
}

// Skip a JSON value (string, number, object, array, true, false, null)
static const char* skipJsonValue(const char* p, const char* end) {
    p = skipWs(p, end);
    if (p >= end) return p;
    if (*p == '"') {
        p++;
        while (p < end && *p != '"') {
            if (*p == '\\') p++;
            p++;
        }
        if (p < end) p++;
        return p;
    }
    if (*p == '{') {
        int depth = 1; p++;
        while (p < end && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') { p++; while (p < end && *p != '"') { if (*p == '\\') p++; p++; } }
            p++;
        }
        return p;
    }
    if (*p == '[') {
        int depth = 1; p++;
        while (p < end && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') { p++; while (p < end && *p != '"') { if (*p == '\\') p++; p++; } }
            p++;
        }
        return p;
    }
    // number, true, false, null
    while (p < end && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    return p;
}

}  // namespace

// ─── Load from tokenizer.json + config ───────────────────────────────────────

bool OnnxTokenizer::load(const std::string& modelDir) {
    // Build byte tables
    build_byte_tables(unicode_to_byte_, byte_to_unicode_);

    // 1. Fast-parse tokenizer.json
    std::string tokPath = (fs::path(modelDir) / "tokenizer.json").string();
    std::ifstream tokFile(tokPath);
    if (!tokFile.is_open()) {
        fprintf(stderr, "Failed to open: %s\n", tokPath.c_str());
        return false;
    }
    std::string tokStr((std::istreambuf_iterator<char>(tokFile)),
                        std::istreambuf_iterator<char>());
    tokFile.close();

    const char* p = tokStr.data();
    const char* end = p + tokStr.size();

    // Find "model" key at top level
    // Strategy: scan for "\"vocab\"" and "\"merges\"" within the "model" object
    // These are the only two things we need.

    // Find "vocab": { ... }
    {
        const char* vocabKey = strstr(p, "\"vocab\"");
        if (!vocabKey) {
            fprintf(stderr, "tokenizer.json: 'vocab' key not found\n");
            return false;
        }
        const char* vp = vocabKey + 7; // past "vocab"
        vp = skipWs(vp, end);
        if (vp < end && *vp == ':') vp++;
        vp = skipWs(vp, end);
        if (vp >= end || *vp != '{') {
            fprintf(stderr, "tokenizer.json: 'vocab' is not an object\n");
            return false;
        }
        vp++; // past '{'

        // Parse vocab entries: "token": id, ...
        // First pass: count entries to size vocab array
        size_t maxId = 0;
        size_t count = 0;
        const char* vp_save = vp;
        while (vp < end) {
            vp = skipWs(vp, end);
            if (*vp == '}') break;
            if (*vp == ',') { vp++; continue; }
            // Parse key string (skip it)
            std::string key;
            vp = parseJsonString(vp, end, key);
            vp = skipWs(vp, end);
            if (vp < end && *vp == ':') vp++;
            vp = skipWs(vp, end);
            // Parse number value
            long id = strtol(vp, (char**)&vp, 10);
            if ((size_t)id > maxId) maxId = (size_t)id;
            count++;
        }

        vocab.resize(maxId + 1);
        token_to_id.reserve(count);

        // Second pass: actually store entries
        vp = vp_save;
        while (vp < end) {
            vp = skipWs(vp, end);
            if (*vp == '}') break;
            if (*vp == ',') { vp++; continue; }
            std::string key;
            vp = parseJsonString(vp, end, key);
            vp = skipWs(vp, end);
            if (vp < end && *vp == ':') vp++;
            vp = skipWs(vp, end);
            int32_t id = (int32_t)strtol(vp, (char**)&vp, 10);
            if (id >= 0 && (size_t)id < vocab.size()) {
                vocab[id] = key;
                token_to_id[key] = id;
            }
        }
        printf("  Vocab: %zu tokens (max_id=%zu)\n", count, maxId);
    }

    // Find "merges": [ ... ]
    // Merges can be either ["a b", ...] (string format) or [["a","b"], ...] (pair format)
    {
        const char* mergesKey = strstr(p, "\"merges\"");
        if (mergesKey) {
            const char* mp = mergesKey + 8;
            mp = skipWs(mp, end);
            if (mp < end && *mp == ':') mp++;
            mp = skipWs(mp, end);
            if (mp < end && *mp == '[') {
                mp++; // past '['
                int32_t rank = 0;
                while (mp < end) {
                    mp = skipWs(mp, end);
                    if (mp >= end || *mp == ']') break;
                    if (*mp == ',') { mp++; continue; }
                    if (*mp == '"') {
                        // String format: "a b"
                        std::string mergeStr;
                        mp = parseJsonString(mp, end, mergeStr);
                        if (mergeStr.empty()) break;
                        merge_rank[mergeStr] = rank++;
                    } else if (*mp == '[') {
                        // Pair format: ["a", "b"]
                        mp++; // past '['
                        mp = skipWs(mp, end);
                        std::string a, b;
                        mp = parseJsonString(mp, end, a);
                        mp = skipWs(mp, end);
                        if (mp < end && *mp == ',') mp++;
                        mp = skipWs(mp, end);
                        mp = parseJsonString(mp, end, b);
                        mp = skipWs(mp, end);
                        if (mp < end && *mp == ']') mp++;
                        if (!a.empty() && !b.empty())
                            merge_rank[a + " " + b] = rank++;
                    } else {
                        break;  // unexpected token
                    }
                }
            }
        }
    }

    // Find "added_tokens": [ {"id": N, "content": "...", ...}, ... ]
    {
        const char* addedKey = strstr(p, "\"added_tokens\"");
        if (addedKey) {
            const char* ap = addedKey + 14;
            ap = skipWs(ap, end);
            if (ap < end && *ap == ':') ap++;
            ap = skipWs(ap, end);
            if (ap < end && *ap == '[') {
                ap++; // past '['
                // Each entry is a JSON object { "id": N, "content": "..." , ... }
                while (ap < end) {
                    ap = skipWs(ap, end);
                    if (ap >= end || *ap == ']') break;
                    if (*ap == ',') { ap++; continue; }
                    if (*ap != '{') break;
                    ap++; // past '{'
                    int32_t id = -1;
                    std::string content;
                    while (ap < end) {
                        ap = skipWs(ap, end);
                        if (ap >= end || *ap == '}') { if (ap < end) ap++; break; }
                        if (*ap == ',') { ap++; continue; }
                        if (*ap != '"') { ap = skipJsonValue(ap, end); continue; }
                        std::string key;
                        ap = parseJsonString(ap, end, key);
                        ap = skipWs(ap, end);
                        if (ap < end && *ap == ':') ap++;
                        ap = skipWs(ap, end);
                        if (key == "id") {
                            id = (int32_t)strtol(ap, (char**)&ap, 10);
                        } else if (key == "content") {
                            ap = parseJsonString(ap, end, content);
                        } else {
                            ap = skipJsonValue(ap, end);
                        }
                    }
                    if (id >= 0 && !content.empty()) {
                        added_tokens[content] = id;
                    }
                }
            }
        }
        // Sort added tokens by length (longest first) for greedy matching
        for (auto& [tok, id] : added_tokens) {
            added_tokens_sorted.push_back(tok);
        }
        std::sort(added_tokens_sorted.begin(), added_tokens_sorted.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
    }

    // 2. Parse genai_config.json or config.json for special tokens
    std::string cfgPath = (fs::path(modelDir) / "genai_config.json").string();
    if (!fs::exists(cfgPath))
        cfgPath = (fs::path(modelDir) / "config.json").string();
    std::ifstream cfgFile(cfgPath);
    if (cfgFile.is_open()) {
        std::string cfgStr((std::istreambuf_iterator<char>(cfgFile)),
                            std::istreambuf_iterator<char>());
        cfgFile.close();
        auto cfgJson = json_parse(cfgStr);
        // genai_config.json nests under "model", config.json is top-level
        auto& root = cfgJson.has("model") ? cfgJson["model"] : cfgJson;

        if (root.has("eos_token_id")) {
            auto& eos = root["eos_token_id"];
            if (eos.is_array())
                eos_token_id = eos[0].as_int();
            else
                eos_token_id = eos.as_int();
        }
        if (root.has("bos_token_id"))
            bos_token_id = root["bos_token_id"].as_int();
    }

    printf("  Tokenizer: %zu tokens, %zu merges, %zu added, EOS=%d\n",
           vocab.size(), merge_rank.size(), added_tokens.size(), eos_token_id);

    return true;
}

// ─── Decode ──────────────────────────────────────────────────────────────────

std::string OnnxTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab.size())
        return "";
    const std::string& tok = vocab[token_id];
    if (tok.size() >= 2 && tok[0] == '<' && tok.back() == '>')
        return tok;
    return bpe_token_to_bytes(tok, unicode_to_byte_);
}

std::string OnnxTokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    for (auto id : token_ids)
        result += decode_token(id);
    return result;
}

// ─── Encode ──────────────────────────────────────────────────────────────────

std::vector<int32_t> OnnxTokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    // 0. Split text on added tokens (special tokens like <|im_start|>, <think>)
    //    These must be matched as exact strings before BPE encoding.
    if (!added_tokens_sorted.empty()) {
        // Split text into segments: alternating [regular_text, special_token, ...]
        struct Segment { std::string text; int32_t tokenId; bool isSpecial; };
        std::vector<Segment> segments;

        size_t pos = 0;
        while (pos < text.size()) {
            // Try to match any added token at current position (longest first)
            bool matched = false;
            for (auto& tok : added_tokens_sorted) {
                if (pos + tok.size() <= text.size() &&
                    text.compare(pos, tok.size(), tok) == 0) {
                    segments.push_back({tok, added_tokens.at(tok), true});
                    pos += tok.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Accumulate regular text
                if (segments.empty() || segments.back().isSpecial)
                    segments.push_back({"", -1, false});
                segments.back().text += text[pos];
                pos++;
            }
        }

        // Encode each segment
        std::vector<int32_t> ids;
        for (auto& seg : segments) {
            if (seg.isSpecial) {
                ids.push_back(seg.tokenId);
            } else if (!seg.text.empty()) {
                auto subIds = encodeBpe(seg.text);
                ids.insert(ids.end(), subIds.begin(), subIds.end());
            }
        }
        return ids;
    }

    // No added tokens — just BPE encode the whole string
    return encodeBpe(text);
}

std::vector<int32_t> OnnxTokenizer::encodeBpe(const std::string& text) const {
    if (text.empty()) return {};

    // 1. Convert bytes to BPE string representation
    std::string bpe_text = bytes_to_bpe_string(text, byte_to_unicode_);

    // 2. Split into characters
    auto pieces = split_to_chars(bpe_text);

    // 3. Iteratively merge the best pair
    while (pieces.size() > 1) {
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
        if (best_idx < 0) break;

        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }

    // 4. Convert to token IDs
    std::vector<int32_t> ids;
    ids.reserve(pieces.size());
    for (auto& p : pieces) {
        auto it = token_to_id.find(p);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            for (uint8_t b : p) {
                uint32_t cp = byte_to_unicode_[b];
                std::string single;
                if (cp < 0x80) single.push_back((char)cp);
                else if (cp < 0x800) {
                    single.push_back((char)(0xC0 | (cp >> 6)));
                    single.push_back((char)(0x80 | (cp & 0x3F)));
                } else {
                    single.push_back((char)(0xE0 | (cp >> 12)));
                    single.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    single.push_back((char)(0x80 | (cp & 0x3F)));
                }
                auto sit = token_to_id.find(single);
                if (sit != token_to_id.end())
                    ids.push_back(sit->second);
            }
        }
    }

    return ids;
}
