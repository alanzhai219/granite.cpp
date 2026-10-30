#include "modeling.hpp"
#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace granite {

GraniteModel::GraniteModel(GraniteWeights weights)
    : weights_(std::move(weights)) {
    kv_cache_ = KVCache(weights_.config.num_hidden_layers,
                        weights_.config.num_key_value_heads,
                        weights_.config.head_dim,
                        weights_.config.max_position_embeddings);
}

std::vector<float> GraniteModel::forward(const std::vector<int>& tokens, size_t n_past) {
    if (tokens.empty()) throw std::invalid_argument("forward: tokens cannot be empty");
    if (n_past != kv_cache_.current_len()) {
        throw std::runtime_error("forward: n_past does not match KV cache length");
    }

    Tensor x = ops::token_embed(weights_.wte, tokens);
    for (float& value : x.data()) value *= weights_.config.embedding_multiplier;
    for (size_t layer_idx = 0; layer_idx < weights_.config.num_hidden_layers; ++layer_idx) {
        transformer_layer(layer_idx, x, n_past);
    }

    Tensor normed = ops::rms_norm(x, weights_.final_norm_w, weights_.config.rms_norm_eps);
    const size_t seq = x.dim(0);
    const size_t hidden = x.dim(1);
    Tensor logits({weights_.config.vocab_size});
    for (size_t i = 0; i < weights_.config.vocab_size; ++i) {
        float value = 0.0F;
        for (size_t j = 0; j < hidden; ++j) {
            value += weights_.wte.at(i, j) * normed.at(seq - 1, j);
        }
        logits.at(i) = value / weights_.config.logits_scaling;
    }
    return std::vector<float>(logits.ptr(), logits.ptr() + logits.numel());
}

void GraniteModel::transformer_layer(size_t layer_id, Tensor& x, size_t n_past) {
    attention_layer(layer_id, x, n_past);
    moe_layer(layer_id, x);
}

void GraniteModel::attention_layer(size_t layer_id, Tensor& x, size_t n_past) {
    const auto& layer = weights_.layers.at(layer_id);
    const size_t seq = x.dim(0);
    const size_t hidden = x.dim(1);
    Tensor normed = ops::rms_norm(x, layer.ln1_w, weights_.config.rms_norm_eps);

    Tensor q = ops::matmul_2d(normed, layer.q_proj_w);
    Tensor k = ops::matmul_2d(normed, layer.k_proj_w);
    Tensor v = ops::matmul_2d(normed, layer.v_proj_w);

    Tensor q_heads = ops::split_heads(q, weights_.config.num_attention_heads, weights_.config.head_dim);
    Tensor k_heads = ops::split_heads(k, weights_.config.num_key_value_heads, weights_.config.head_dim);
    Tensor v_heads = ops::split_heads(v, weights_.config.num_key_value_heads, weights_.config.head_dim);

    ops::apply_rope(q_heads, n_past, weights_.config.rope_theta);
    ops::apply_rope(k_heads, n_past, weights_.config.rope_theta);

    auto& layer_cache = kv_cache_.layers[layer_id];
    size_t k_flat_size = weights_.config.num_key_value_heads * weights_.config.head_dim;
    std::vector<float> k_flat(seq * k_flat_size, 0.0F);
    std::vector<float> v_flat(seq * k_flat_size, 0.0F);
    for (size_t s = 0; s < seq; ++s) {
        for (size_t h = 0; h < weights_.config.num_key_value_heads; ++h) {
            for (size_t d = 0; d < weights_.config.head_dim; ++d) {
                const size_t idx = s * k_flat_size + h * weights_.config.head_dim + d;
                k_flat[idx] = k_heads.at(h, s, d);
                v_flat[idx] = v_heads.at(h, s, d);
            }
        }
    }
    layer_cache.append(k_flat.data(), v_flat.data(), seq);

    const size_t q_per_kv = weights_.config.num_attention_heads / weights_.config.num_key_value_heads;
    Tensor attn_out({seq, hidden});
    for (size_t s = 0; s < seq; ++s) {
        for (size_t hq = 0; hq < weights_.config.num_attention_heads; ++hq) {
            const size_t hk = hq / q_per_kv;
            std::vector<float> scores(layer_cache.current_len, 0.0F);
            for (size_t t = 0; t < layer_cache.current_len; ++t) {
                const size_t src_k = (hk * layer_cache.max_cache_len + t) * weights_.config.head_dim;
                float dot = 0.0F;
                for (size_t d = 0; d < weights_.config.head_dim; ++d) dot += q_heads.at(hq, s, d) * layer_cache.k_cache[src_k + d];
                scores[t] = dot * weights_.config.attention_multiplier;
                if (t > n_past + s) scores[t] = -std::numeric_limits<float>::infinity();
            }
            const auto probs = ops::softmax_vector(scores);
            for (size_t d = 0; d < weights_.config.head_dim; ++d) {
                float result = 0.0F;
                for (size_t t = 0; t < layer_cache.current_len; ++t) {
                    const size_t src_v = (hk * layer_cache.max_cache_len + t) * weights_.config.head_dim;
                    result += probs[t] * layer_cache.v_cache[src_v + d];
                }
                attn_out.at(s, hq * weights_.config.head_dim + d) = result;
            }
        }
    }

    Tensor projected = ops::matmul_2d(attn_out, layer.o_proj_w);
    for (size_t i = 0; i < x.dim(0); ++i) {
        for (size_t j = 0; j < x.dim(1); ++j) {
            x.at(i, j) += projected.at(i, j) * weights_.config.residual_multiplier;
        }
    }
}

void GraniteModel::moe_layer(size_t layer_id, Tensor& x) {
    const auto& layer = weights_.layers.at(layer_id);
    Tensor normed = ops::rms_norm(x, layer.ln2_w, weights_.config.rms_norm_eps);
    const size_t seq = x.dim(0);
    const size_t hidden = x.dim(1);
    const size_t experts = layer.moe_router_w.dim(0);
    std::vector<float> combined(hidden, 0.0F);
    for (size_t s = 0; s < seq; ++s) {
        std::vector<float> logits(experts, 0.0F);
        for (size_t e = 0; e < experts; ++e) {
            float sum = 0.0F;
            for (size_t i = 0; i < hidden; ++i) sum += normed.at(s, i) * layer.moe_router_w.at(e, i);
            logits[e] = sum;
        }
        const auto chosen = ops::topk_indices(logits, weights_.config.num_experts_per_tok);
        std::vector<float> selected_logits;
        for (int expert_idx : chosen) selected_logits.push_back(logits[expert_idx]);
        const auto routing_weights = ops::softmax_vector(selected_logits);
        for (size_t i = 0; i < hidden; ++i) combined[i] = 0.0F;
        for (size_t choice = 0; choice < chosen.size(); ++choice) {
            const size_t expert_idx = static_cast<size_t>(chosen[choice]);
            std::vector<float> expert_hidden(weights_.config.intermediate_size, 0.0F);
            for (size_t j = 0; j < expert_hidden.size(); ++j) {
                float gate = 0.0F, up = 0.0F;
                const float* input = normed.ptr() + s * hidden;
                const float* gate_w = layer.moe_input_w.ptr() + (expert_idx * 2 * expert_hidden.size() + j) * hidden;
                const float* up_w = gate_w + expert_hidden.size() * hidden;
                for (size_t i = 0; i < hidden; ++i) {
                    gate += input[i] * gate_w[i];
                    up += input[i] * up_w[i];
                }
                expert_hidden[j] = ops::silu(gate) * up;
            }
            for (size_t j = 0; j < hidden; ++j) {
                float sum = 0.0F;
                const float* down = layer.moe_output_w.ptr() + (expert_idx * hidden + j) * expert_hidden.size();
                for (size_t i = 0; i < expert_hidden.size(); ++i) sum += expert_hidden[i] * down[i];
                combined[j] += routing_weights[choice] * sum;
            }
        }
        for (size_t i = 0; i < hidden; ++i) x.at(s, i) += combined[i] * weights_.config.residual_multiplier;
    }
}

std::string GraniteModel::generate(const tk::Tokenizer& tokenizer,
                                  const std::string& prompt,
                                  int max_tokens,
                                  float temperature,
                                  int top_k,
                                  uint64_t seed) {
    if (max_tokens < 0) throw std::invalid_argument("max_tokens must be >= 0");
    rng_.seed(seed);
    kv_cache_.reset();

    std::vector<int> ids = tokenizer.encode(prompt);
    if (ids.empty()) ids.push_back(0);
    if (max_tokens > 0) ids.reserve(ids.size() + static_cast<size_t>(max_tokens));

    std::vector<float> logits = forward(ids, 0);
    const int end_id = 0;
    for (int step = 0; step < max_tokens; ++step) {
        int next = 0;
        if (temperature <= 0.0F || top_k <= 0) {
            auto it = std::max_element(logits.begin(), logits.end());
            next = static_cast<int>(std::distance(logits.begin(), it));
        } else {
            const size_t count = std::min<std::size_t>(top_k, logits.size());
            std::vector<int> candidates(logits.size());
            std::iota(candidates.begin(), candidates.end(), 0);
            std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::vector<double> weights;
            weights.reserve(count);
            const float max_logit = logits[candidates.front()];
            for (size_t i = 0; i < count; ++i) {
                weights.push_back(std::exp((logits[candidates[i]] - max_logit) / temperature));
            }
            std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
            next = candidates[dist(rng_)];
        }
        if (next == end_id) break;
        ids.push_back(next);
        if (step + 1 < max_tokens) logits = forward({next}, kv_cache_.current_len());
    }
    return tokenizer.decode(ids);
}

}  // namespace granite
