#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace granite::ops {

Tensor token_embed(const Tensor& wte, const std::vector<int>& ids) {
    if (wte.ndims() != 2) throw std::invalid_argument("token_embed: expected [vocab, hidden]");
    const size_t hidden = wte.dim(1);
    Tensor out({ids.size(), hidden});
    for (size_t i = 0; i < ids.size(); ++i) {
        const int token_id = ids[i];
        if (token_id < 0 || static_cast<size_t>(token_id) >= wte.dim(0)) {
            throw std::out_of_range("token_embed: token id out of range");
        }
        for (size_t j = 0; j < hidden; ++j) {
            out.at(i, j) = wte.at(static_cast<size_t>(token_id), j);
        }
    }
    return out;
}

Tensor matmul_2d(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.ndims() != 2 || rhs.ndims() != 2 || lhs.dim(1) != rhs.dim(1)) {
        throw std::invalid_argument("matmul_2d: incompatible shape");
    }
    const size_t rows = lhs.dim(0);
    const size_t cols = rhs.dim(0);
    const size_t inner = lhs.dim(1);
    Tensor out({rows, cols});
    for (size_t i = 0; i < rows; ++i) {
        const float* input = lhs.ptr() + i * inner;
        for (size_t j = 0; j < cols; ++j) {
            const float* weight = rhs.ptr() + j * inner;
            float value = 0.0F;
            for (size_t k = 0; k < inner; ++k) {
                value += input[k] * weight[k];
            }
            out.ptr()[i * cols + j] = value;
        }
    }
    return out;
}

Tensor rms_norm(const Tensor& input, const Tensor& gamma, float eps) {
    if (input.ndims() != 2) throw std::invalid_argument("rms_norm: expected [seq, hidden]");
    const size_t hidden = input.dim(1);
    if (gamma.ndims() != 1 || gamma.dim(0) != hidden) throw std::invalid_argument("rms_norm: weight shape mismatch");
    Tensor out({input.dim(0), hidden});
    for (size_t row = 0; row < input.dim(0); ++row) {
        float variance = 0.0F;
        for (size_t j = 0; j < hidden; ++j) {
            const float value = input.at(row, j);
            variance += value * value;
        }
        variance /= static_cast<float>(hidden);
        const float inv_std = 1.0F / std::sqrt(variance + eps);
        for (size_t j = 0; j < hidden; ++j) {
            out.at(row, j) = input.at(row, j) * inv_std * gamma.at(j);
        }
    }
    return out;
}

Tensor split_heads(const Tensor& x, size_t num_heads, size_t head_dim) {
    if (x.ndims() != 2 || x.dim(1) != num_heads * head_dim) {
        throw std::invalid_argument("split_heads: incompatible shape");
    }
    Tensor out({num_heads, x.dim(0), head_dim});
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < x.dim(0); ++s) {
            for (size_t d = 0; d < head_dim; ++d) {
                out.at(h, s, d) = x.at(s, h * head_dim + d);
            }
        }
    }
    return out;
}

float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

std::vector<float> softmax_vector(const std::vector<float>& values) {
    if (values.empty()) return {};
    std::vector<float> out(values.size());
    float max_v = *std::max_element(values.begin(), values.end());
    float sum = 0.0F;
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = std::exp(values[i] - max_v);
        sum += out[i];
    }
    const float inv_sum = 1.0F / sum;
    for (size_t i = 0; i < values.size(); ++i) out[i] *= inv_sum;
    return out;
}

std::vector<int> topk_indices(const std::vector<float>& values, size_t k) {
    if (values.empty()) return {};
    std::vector<size_t> ids(values.size());
    std::iota(ids.begin(), ids.end(), 0);
    const size_t limit = std::min(k, values.size());
    std::partial_sort(ids.begin(), ids.begin() + limit, ids.end(),
                      [&](size_t a, size_t b) { return values[a] > values[b]; });
    std::vector<int> selected;
    selected.reserve(limit);
    for (size_t i = 0; i < limit; ++i) selected.push_back(static_cast<int>(ids[i]));
    return selected;
}

void apply_rope(Tensor& heads, size_t pos_offset, float theta) {
    if (heads.ndims() != 3) return;
    const size_t num_heads = heads.dim(0);
    const size_t seq = heads.dim(1);
    const size_t head_dim = heads.dim(2);
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq; ++s) {
            const size_t position = pos_offset + s;
            const size_t half = head_dim / 2;
            for (size_t d = 0; d < half; ++d) {
                const float angle = static_cast<float>(position) / std::pow(theta, static_cast<float>(2 * d) / static_cast<float>(head_dim));
                const float cos_v = std::cos(angle);
                const float sin_v = std::sin(angle);
                const float x0 = heads.at(h, s, d);
                const float x1 = heads.at(h, s, d + half);
                heads.at(h, s, d) = x0 * cos_v - x1 * sin_v;
                heads.at(h, s, d + half) = x0 * sin_v + x1 * cos_v;
            }
        }
    }
}

}  // namespace granite::ops
