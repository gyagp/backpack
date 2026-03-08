#include "json_parser.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

struct Parser {
    const std::string& s;
    size_t pos = 0;

    void skipWs() { while (pos < s.size() && isspace(s[pos])) pos++; }
    char peek()   { skipWs(); return pos < s.size() ? s[pos] : 0; }
    char next()   { skipWs(); return pos < s.size() ? s[pos++] : 0; }

    void expect(char c) {
        char g = next();
        if (g != c) {
            fprintf(stderr, "JSON: expected '%c', got '%c' at pos %zu\n",
                    c, g, pos);
            exit(1);
        }
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') {
                pos++;
                switch (s[pos]) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '/': out += '/'; break;
                    default: out += s[pos]; break;
                }
            } else {
                out += s[pos];
            }
            pos++;
        }
        expect('"');
        return out;
    }

    JsonValue parseValue() {
        char c = peek();
        if (c == '"') return JsonValue{parseString()};
        if (c == '{') return JsonValue{parseObject()};
        if (c == '[') return JsonValue{parseArray()};
        if (c == 't') { pos += 4; return JsonValue{true}; }
        if (c == 'f') { pos += 5; return JsonValue{false}; }
        if (c == 'n') { pos += 4; return JsonValue{nullptr}; }
        // Number
        skipWs();
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() && (isdigit(s[pos]) || s[pos] == '.' ||
               s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-'))
            pos++;
        return JsonValue{std::stod(s.substr(start, pos - start))};
    }

    JsonObject parseObject() {
        expect('{');
        JsonObject obj;
        if (peek() == '}') { next(); return obj; }
        while (true) {
            auto key = parseString();
            expect(':');
            obj[key] = parseValue();
            if (peek() == ',') { next(); continue; }
            break;
        }
        expect('}');
        return obj;
    }

    JsonArray parseArray() {
        expect('[');
        JsonArray arr;
        if (peek() == ']') { next(); return arr; }
        while (true) {
            arr.push_back(parseValue());
            if (peek() == ',') { next(); continue; }
            break;
        }
        expect(']');
        return arr;
    }
};

JsonValue json_parse(const std::string& text) {
    Parser p{text};
    return p.parseValue();
}
