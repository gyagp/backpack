#include "../src/chat_template.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    // Test system + user + assistant messages
    {
        std::vector<ChatMessage> messages = {
            {"system", "You are a helpful assistant."},
            {"user", "Hello!"},
            {"assistant", "Hi there!"},
        };

        std::string result = format_chat(messages, false);
        std::string expected =
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello!<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Hi there!<|im_end|>\n";
        assert(result == expected);
    }

    // Test with generation prompt
    {
        std::vector<ChatMessage> messages = {
            {"system", "You are a helpful assistant."},
            {"user", "What is 2+2?"},
        };

        std::string result = format_chat(messages, true);
        std::string expected =
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n";
        assert(result == expected);
    }

    // Test empty messages
    {
        std::vector<ChatMessage> messages;
        std::string result = format_chat(messages, false);
        assert(result.empty());
    }

    // Test empty messages with generation prompt
    {
        std::vector<ChatMessage> messages;
        std::string result = format_chat(messages, true);
        assert(result == "<|im_start|>assistant\n");
    }

    std::cout << "All chat_template tests passed." << std::endl;
    return 0;
}
