#pragma once

#include <cstddef>
#include <vector>

#include "tensor.hpp"

namespace granite::ops {

// Stateless tensor/math operators. Decoder-layer orchestration stays in GraniteModel.
Tensor token_embed(const Tensor& embeddings, const std::vector<int>& ids);
Tensor matmul_2d(const Tensor& lhs, const Tensor& rhs);
Tensor rms_norm(const Tensor& input, const Tensor& gamma, float eps);
Tensor split_heads(const Tensor& input, size_t num_heads, size_t head_dim);
void apply_rope(Tensor& heads, size_t pos_offset, float theta);
float silu(float value);
std::vector<float> softmax_vector(const std::vector<float>& values);
std::vector<int> topk_indices(const std::vector<float>& values, size_t k);

}  // namespace granite::ops
