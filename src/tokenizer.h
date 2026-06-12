#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf_parser.h"

struct TokenizerVocab {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
};

inline TokenizerVocab extract_vocab(const GGUFFile& gguf) {
    TokenizerVocab vocab;

    auto get_array = [&](const std::string& key) -> const GGUFArray& {
        auto it = gguf.metadata.find(key);
        if (it == gguf.metadata.end())
            throw std::runtime_error("tokenizer: missing metadata key: " + key);
        if (!std::holds_alternative<GGUFArray>(it->second))
            throw std::runtime_error("tokenizer: expected array for key: " + key);
        return std::get<GGUFArray>(it->second);
    };

    const auto& tokens_arr = get_array("tokenizer.ggml.tokens");
    vocab.tokens.reserve(tokens_arr.values.size());
    for (const auto& v : tokens_arr.values)
        vocab.tokens.push_back(std::get<std::string>(v));

    const auto& scores_arr = get_array("tokenizer.ggml.scores");
    vocab.scores.reserve(scores_arr.values.size());
    for (const auto& v : scores_arr.values)
        vocab.scores.push_back(std::get<float>(v));

    const auto& types_arr = get_array("tokenizer.ggml.token_type");
    vocab.token_types.reserve(types_arr.values.size());
    for (const auto& v : types_arr.values)
        vocab.token_types.push_back(std::get<int32_t>(v));

    auto merges_it = gguf.metadata.find("tokenizer.ggml.merges");
    if (merges_it != gguf.metadata.end() &&
        std::holds_alternative<GGUFArray>(merges_it->second)) {
        const auto& merges_arr = std::get<GGUFArray>(merges_it->second);
        vocab.merges.reserve(merges_arr.values.size());
        for (const auto& v : merges_arr.values)
            vocab.merges.push_back(std::get<std::string>(v));
    }

    return vocab;
}

class BPETokenizer {
public:
    explicit BPETokenizer(const TokenizerVocab& vocab) {
        for (uint32_t i = 0; i < vocab.tokens.size(); ++i) {
            token_to_id_[vocab.tokens[i]] = i;
            id_to_token_.push_back(vocab.tokens[i]);
        }
        for (size_t i = 0; i < vocab.merges.size(); ++i) {
            auto sp = vocab.merges[i].find(' ');
            if (sp == std::string::npos) continue;
            std::string left = vocab.merges[i].substr(0, sp);
            std::string right = vocab.merges[i].substr(sp + 1);
            merge_rank_[{left, right}] = static_cast<int>(i);
        }
    }

    std::vector<uint32_t> encode(const std::string& text) const {
        if (text.empty()) return {};

        std::vector<std::string> symbols;
        for (size_t i = 0; i < text.size(); ++i)
            symbols.push_back(std::string(1, text[i]));

        while (symbols.size() > 1) {
            int best_rank = (std::numeric_limits<int>::max)();
            size_t best_pos = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto it = merge_rank_.find({symbols[i], symbols[i + 1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                    found = true;
                }
            }
            if (!found) break;

            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        std::vector<uint32_t> ids;
        ids.reserve(symbols.size());
        for (const auto& s : symbols) {
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                for (unsigned char c : s) {
                    std::string byte_tok = "<0x" + byte_to_hex(c) + ">";
                    auto bit = token_to_id_.find(byte_tok);
                    if (bit != token_to_id_.end())
                        ids.push_back(bit->second);
                }
            }
        }
        return ids;
    }

    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string result;
        for (uint32_t id : ids) {
            if (id >= id_to_token_.size()) continue;
            const std::string& tok = id_to_token_[id];
            if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' &&
                tok[2] == 'x' && tok[5] == '>') {
                unsigned val = 0;
                if (hex_to_byte(tok[3], tok[4], val))
                    result += static_cast<char>(val);
                else
                    result += tok;
            } else {
                result += tok;
            }
        }
        return result;
    }

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank_;

    static std::string byte_to_hex(unsigned char c) {
        const char* hex = "0123456789ABCDEF";
        return {hex[c >> 4], hex[c & 0xf]};
    }

    static bool hex_to_byte(char h, char l, unsigned& out) {
        auto hv = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
        };
        int hi = hv(h), lo = hv(l);
        if (hi < 0 || lo < 0) return false;
        out = static_cast<unsigned>(hi << 4 | lo);
        return true;
    }
};
