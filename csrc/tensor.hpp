#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace granite {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<size_t> shape)
        : shape_(std::move(shape)) {
        data_.resize(numel(), 0.0F);
    }

    Tensor(std::vector<size_t> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (data_.size() != numel()) {
            throw std::invalid_argument("Tensor: shape and data size mismatch");
        }
    }

    size_t ndims() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t dim(size_t idx) const { return shape_.at(idx); }
    size_t numel() const {
        size_t n = 1;
        for (size_t v : shape_) n *= v;
        return n;
    }

    float* ptr() { return data_.data(); }
    const float* ptr() const { return data_.data(); }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    float& at(size_t i) {
        return data_.at(i);
    }
    const float& at(size_t i) const {
        return data_.at(i);
    }

    float& at(size_t i0, size_t i1) {
        return data_.at(index({i0, i1}));
    }
    const float& at(size_t i0, size_t i1) const {
        return data_.at(index({i0, i1}));
    }

    float& at(size_t i0, size_t i1, size_t i2) {
        return data_.at(index({i0, i1, i2}));
    }
    const float& at(size_t i0, size_t i1, size_t i2) const {
        return data_.at(index({i0, i1, i2}));
    }

    Tensor copy() const {
        return Tensor(shape_, data_);
    }

private:
    size_t index(std::initializer_list<size_t> dims) const {
        if (dims.size() != shape_.size()) {
            throw std::invalid_argument("Tensor: index rank mismatch");
        }
        size_t offset = 0;
        size_t stride = 1;
        auto it = dims.end();
        for (size_t i = shape_.size(); i-- > 0;) {
            --it;
            const size_t dim = shape_[i];
            const size_t idx = *it;
            if (idx >= dim) {
                throw std::out_of_range("Tensor index out of range");
            }
            offset += idx * stride;
            stride *= dim;
        }
        return offset;
    }

    std::vector<size_t> shape_;
    std::vector<float> data_;
};

}  // namespace granite
