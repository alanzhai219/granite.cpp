#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tk {

class Tokenizer {
public:
    Tokenizer(const std::string& encoder_json_path,
              const std::string& vocab_bpe_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& token) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> bpe_ranks_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    std::array<std::string, 256> byte_encoder_;
    std::array<int16_t, 512> byte_decoder_;
};

}  // namespace tk
