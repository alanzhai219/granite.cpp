#include "tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tk {

static size_t utf8_char_length(unsigned char lead_byte) {
    if (lead_byte < 0x80) return 1;
    if (lead_byte >= 0xC2 && lead_byte <= 0xDF) return 2;
    if (lead_byte >= 0xE0 && lead_byte <= 0xEF) return 3;
    if (lead_byte >= 0xF0 && lead_byte <= 0xF4) return 4;
    throw std::runtime_error("invalid UTF-8 leading byte");
}

static std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char lead_byte = static_cast<unsigned char>(s[i]);
        const size_t len = utf8_char_length(lead_byte);
        if (i + len > n) throw std::runtime_error("truncated UTF-8 sequence");
        uint32_t cp = lead_byte;
        if (len == 2) cp &= 0x1F;
        else if (len == 3) cp &= 0x0F;
        else if (len == 4) cp &= 0x07;

        for (size_t k = 1; k < len; ++k) {
            const unsigned char byte = static_cast<unsigned char>(s[i + k]);
            if ((byte & 0xC0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation byte");
            cp = (cp << 6) | (byte & 0x3F);
        }
        if ((len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
            (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
            throw std::runtime_error("invalid UTF-8 codepoint");
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

static void utf8_encode(uint32_t cp, std::string& out) {
    if ((cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        throw std::runtime_error("invalid Unicode codepoint");
    }
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

static bool is_space_cp(uint32_t cp) {
    return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0;
}

static bool is_number_cp(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if (cp >= 0x80 && !is_space_cp(cp)) return true;
    return false;
}

static size_t contraction_length(const std::vector<uint32_t>& codepoints, size_t start) {
    const size_t remaining = codepoints.size() - start;
    if (remaining < 2 || codepoints[start] != '\'') return 0;
    const uint32_t second = codepoints[start + 1];
    if (second == 's' || second == 't' || second == 'm' || second == 'd') return 2;
    if (remaining >= 3) {
        const uint32_t third = codepoints[start + 2];
        if ((second == 'r' && third == 'e') ||
            (second == 'v' && third == 'e') ||
            (second == 'l' && third == 'l')) {
            return 3;
        }
    }
    return 0;
}

static std::vector<std::string> split_utf8_chars(const std::string& text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        const size_t len = utf8_char_length(static_cast<unsigned char>(text[i]));
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

static void build_byte_table(std::array<std::string, 256>& byte_encoder,
                             std::array<int16_t, 512>& byte_decoder) {
    byte_decoder.fill(-1);
    bool printable[256] = {false};
    for (int b = '!'; b <= '~'; ++b) printable[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;

    uint32_t next_codepoint = 256;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = printable[b] ? static_cast<uint32_t>(b) : next_codepoint++;
        byte_encoder[b].clear();
        utf8_encode(cp, byte_encoder[b]);
        byte_decoder[cp] = static_cast<int16_t>(b);
    }
}

static std::string encode_bytes(const std::string& text,
                                const std::array<std::string, 256>& byte_encoder) {
    std::string encoded;
    for (unsigned char byte : text) encoded += byte_encoder[byte];
    return encoded;
}

static std::string decode_bytes(const std::string& encoded,
                                const std::array<int16_t, 512>& byte_decoder) {
    std::string decoded;
    for (uint32_t cp : utf8_decode(encoded)) {
        const int16_t byte = cp < byte_decoder.size() ? byte_decoder[cp] : -1;
        if (byte < 0) throw std::runtime_error("token is not a byte-level sequence");
        decoded.push_back(static_cast<char>(byte));
    }
    return decoded;
}

class EncoderJsonParser {
public:
    explicit EncoderJsonParser(const std::string& text) : text_(text) {}

    std::unordered_map<std::string, int> parse() {
        skip_whitespace();
        expect('{');
        std::unordered_map<std::string, int> result;
        skip_whitespace();
        while (!consume('}')) {
            std::string token = read_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            if (!result.emplace(std::move(token), read_int()).second) {
                throw std::runtime_error("encoder.json: duplicate token");
            }
            skip_whitespace();
            if (consume('}')) break;
            expect(',');
            skip_whitespace();
        }
        skip_whitespace();
        return result;
    }

private:
    const std::string& text_;
    size_t cursor_ = 0;

    bool at_end() const { return cursor_ == text_.size(); }
    char peek() const { return at_end() ? '\0' : text_[cursor_]; }
    bool consume(char expected) {
        if (peek() != expected) return false;
        ++cursor_;
        return true;
    }
    void expect(char expected) {
        if (!consume(expected)) throw std::runtime_error("encoder.json: unexpected character");
    }
    void skip_whitespace() {
        while (peek() == ' ' || peek() == '\n' || peek() == '\t' || peek() == '\r') ++cursor_;
    }
    static int hex_value(char ch) {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    }
    uint32_t read_code_unit() {
        if (text_.size() - cursor_ < 4) throw std::runtime_error("encoder.json: incomplete Unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const int digit = hex_value(text_[cursor_++]);
            if (digit < 0) throw std::runtime_error("encoder.json: invalid Unicode escape");
            value = (value << 4) | static_cast<uint32_t>(digit);
        }
        return value;
    }
    std::string read_string() {
        expect('"');
        std::string value;
        while (!at_end()) {
            const char ch = text_[cursor_++];
            if (ch == '"') return value;
            if (static_cast<unsigned char>(ch) < 0x20) throw std::runtime_error("encoder.json: control character in string");
            if (ch != '\\') {
                value.push_back(ch);
                continue;
            }
            if (at_end()) throw std::runtime_error("encoder.json: incomplete escape");
            switch (text_[cursor_++]) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = read_code_unit();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (!consume('\\') || !consume('u')) throw std::runtime_error("encoder.json: missing surrogate pair");
                        const uint32_t low = read_code_unit();
                        if (low < 0xDC00 || low > 0xDFFF) throw std::runtime_error("encoder.json: invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        throw std::runtime_error("encoder.json: unexpected low surrogate");
                    }
                    utf8_encode(cp, value);
                    break;
                }
                default: throw std::runtime_error("encoder.json: invalid escape");
            }
        }
        throw std::runtime_error("encoder.json: unterminated string");
    }
    int read_int() {
        const bool negative = consume('-');
        if (peek() < '0' || peek() > '9') throw std::runtime_error("encoder.json: expected integer ID");
        uint64_t value = 0;
        while (peek() >= '0' && peek() <= '9') {
            value = value * 10 + static_cast<uint64_t>(text_[cursor_++] - '0');
        }
        if (negative) return -static_cast<int>(value);
        return static_cast<int>(value);
    }
};

static std::unordered_map<std::string, int> parse_encoder_json(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return EncoderJsonParser(buffer.str()).parse();
}

Tokenizer::Tokenizer(const std::string& encoder_json_path,
                     const std::string& vocab_bpe_path) {
    build_byte_table(byte_encoder_, byte_decoder_);
    token_to_id_ = parse_encoder_json(encoder_json_path);
    if (token_to_id_.empty()) throw std::runtime_error("encoder.json: empty vocabulary");

    int max_id = -1;
    for (const auto& kv : token_to_id_) {
        if (kv.second < 0) throw std::runtime_error("encoder.json: negative token ID");
        max_id = std::max(max_id, kv.second);
    }
    id_to_token_.resize(static_cast<size_t>(max_id) + 1);
    for (const auto& kv : token_to_id_) {
        if (!id_to_token_[static_cast<size_t>(kv.second)].empty()) {
            throw std::runtime_error("encoder.json: duplicate token ID");
        }
        id_to_token_[static_cast<size_t>(kv.second)] = kv.first;
    }

    std::ifstream bpe_file(vocab_bpe_path, std::ios::binary);
    if (!bpe_file) throw std::runtime_error("cannot open " + vocab_bpe_path);
    std::string line;
    std::getline(bpe_file, line);  // skip header
    int rank = 0;
    while (std::getline(bpe_file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        bpe_ranks_[line] = rank++;
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<uint32_t> cps = utf8_decode(text);
    std::vector<std::string> pieces;
    auto emit = [&](size_t start, size_t end) {
        std::string piece;
        for (size_t i = start; i < end; ++i) utf8_encode(cps[i], piece);
        pieces.push_back(piece);
    };

    size_t i = 0;
    while (i < cps.size()) {
        const size_t contraction = contraction_length(cps, i);
        if (contraction > 0) {
            emit(i, i + contraction);
            i += contraction;
            continue;
        }

        bool lead_space = (cps[i] == ' ');
        size_t base = lead_space ? i + 1 : i;

        if (base < cps.size() && is_letter_cp(cps[base])) {
            size_t j = base;
            while (j < cps.size() && is_letter_cp(cps[j])) ++j;
            emit(i, j);
            i = j;
            continue;
        }
        if (base < cps.size() && is_number_cp(cps[base])) {
            size_t j = base;
            while (j < cps.size() && is_number_cp(cps[j])) ++j;
            emit(i, j);
            i = j;
            continue;
        }
        if (base < cps.size() && !is_space_cp(cps[base]) && !is_letter_cp(cps[base]) && !is_number_cp(cps[base])) {
            size_t j = base;
            while (j < cps.size() && !is_space_cp(cps[j]) && !is_letter_cp(cps[j]) && !is_number_cp(cps[j])) ++j;
            emit(i, j);
            i = j;
            continue;
        }
        if (is_space_cp(cps[i])) {
            size_t e = i;
            while (e < cps.size() && is_space_cp(cps[e])) ++e;
            size_t match_end = e;
            if (e < cps.size() && e - i >= 1) {
                match_end = e - 1;
                if (match_end <= i) match_end = i + 1;
            }
            if (match_end == i) match_end = i + 1;
            emit(i, match_end);
            i = match_end;
            continue;
        }

        emit(i, i + 1);
        ++i;
    }
    return pieces;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    auto cached = bpe_cache_.find(token);
    if (cached != bpe_cache_.end()) return cached->second;

    std::vector<std::string> word = split_utf8_chars(token);
    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string pair = word[i] + " " + word[i + 1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                found = true;
            }
        }
        if (!found) break;

        const std::string first = word[best_idx];
        const std::string second = word[best_idx + 1];
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                ++i;
            }
        }
        word = std::move(new_word);
    }

    bpe_cache_.emplace(token, word);
    return word;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const std::string& piece : pretokenize(text)) {
        const std::string encoded = encode_bytes(piece, byte_encoder_);
        for (const std::string& sym : bpe(encoded)) {
            auto it = token_to_id_.find(sym);
            if (it == token_to_id_.end()) throw std::runtime_error("BPE produced unknown token: " + sym);
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string encoded;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() || id_to_token_[static_cast<size_t>(id)].empty()) {
            throw std::out_of_range("token ID is not in the vocabulary");
        }
        encoded += id_to_token_[static_cast<size_t>(id)];
    }
    return decode_bytes(encoded, byte_decoder_);
}

}  // namespace tk
