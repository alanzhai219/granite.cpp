#pragma once

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

namespace granite {

struct KVCacheLayer {
    size_t head_dim = 0;
    size_t num_heads = 0;
    size_t max_cache_len = 0;
    size_t current_len = 0;
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    explicit KVCacheLayer(size_t num_heads_, size_t head_dim_, size_t max_cache_len_)
        : head_dim(head_dim_), num_heads(num_heads_), max_cache_len(max_cache_len_) {
        k_cache.resize(num_heads * max_cache_len * head_dim);
        v_cache.resize(num_heads * max_cache_len * head_dim);
    }

    void reset() {
        current_len = 0;
    }

    void append(const float* new_k, const float* new_v, size_t seq_len) {
        if (current_len + seq_len > max_cache_len) {
            throw std::runtime_error("KV cache overflow");
        }
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t src = s * num_heads * head_dim + h * head_dim;
                const size_t dst = (h * max_cache_len + current_len + s) * head_dim;
                std::memcpy(k_cache.data() + dst, new_k + src, head_dim * sizeof(float));
                std::memcpy(v_cache.data() + dst, new_v + src, head_dim * sizeof(float));
            }
        }
        current_len += seq_len;
    }
};

struct KVCache {
    KVCache() = default;
    KVCache(size_t num_layers, size_t num_heads, size_t head_dim, size_t max_cache_len)
        : layers(num_layers, KVCacheLayer(num_heads, head_dim, max_cache_len)) {}

    void reset() {
        for (auto& layer : layers) layer.reset();
    }

    size_t current_len() const {
        return layers.empty() ? 0 : layers.front().current_len;
    }

    std::vector<KVCacheLayer> layers;
};

class GraniteModel {
public:
    explicit GraniteModel(GraniteWeights weights);

    std::vector<float> forward(const std::vector<int>& tokens, size_t n_past);
    std::string generate(const tk::Tokenizer& tokenizer,
                         const std::string& prompt,
                         int max_tokens,
                         float temperature,
                         int top_k,
                         uint64_t seed);

private:
    void transformer_layer(size_t layer_id, Tensor& x, size_t n_past);
    void attention_layer(size_t layer_id, Tensor& x, size_t n_past);
    void moe_layer(size_t layer_id, Tensor& x);

    GraniteWeights weights_;
    KVCache kv_cache_;
    std::mt19937_64 rng_;
};

}  // namespace granite
