#pragma once

#include <string>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string content;
};

inline std::string format_chat(const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt = true) {
    std::string result;
    for (const auto& msg : messages) {
        result += "<|im_start|>";
        result += msg.role;
        result += "\n";
        result += msg.content;
        result += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }
    return result;
}
