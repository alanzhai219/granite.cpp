#pragma once

#include <cstddef>

namespace granite {

struct GraniteConfig {
    size_t vocab_size = 49155;
    size_t hidden_size = 1536;
    size_t num_hidden_layers = 32;
    size_t num_attention_heads = 24;
    size_t num_key_value_heads = 8;
    size_t max_position_embeddings = 4096;
    size_t intermediate_size = 512;
    size_t num_local_experts = 40;
    size_t num_experts_per_tok = 8;
    size_t head_dim = 64;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 10000.0F;
    float attention_multiplier = 0.015625F;
    float embedding_multiplier = 12.0F;
    float residual_multiplier = 0.22F;
    float logits_scaling = 6.0F;
};

}  // namespace granite
