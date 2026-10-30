#include "weights.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace granite {
namespace {

class JsonCursor {
public:
    explicit JsonCursor(const std::string& text) : text_(text) {}

    void whitespace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' || text_[pos_] == '\t')) ++pos_;
    }
    char peek() {
        whitespace();
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }
    void expect(char ch) {
        whitespace();
        if (pos_ >= text_.size() || text_[pos_] != ch) throw std::runtime_error("safetensors: malformed JSON header");
        ++pos_;
    }

    std::string string() {
        expect('"');
        std::string result;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            if (text_[pos_] == '\\') {
                if (++pos_ >= text_.size()) throw std::runtime_error("safetensors: bad JSON escape");
            }
            result.push_back(text_[pos_++]);
        }
        if (pos_ >= text_.size()) throw std::runtime_error("safetensors: unterminated JSON string");
        ++pos_;
        return result;
    }

    uint64_t integer() {
        whitespace();
        const size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        if (start == pos_) throw std::runtime_error("safetensors: expected integer");
        return std::stoull(text_.substr(start, pos_ - start));
    }

    std::vector<size_t> array() {
        std::vector<size_t> result;
        expect('[');
        while (peek() != ']') {
            result.push_back(static_cast<size_t>(integer()));
            if (peek() == ',') { expect(','); continue; }
            if (peek() != ']') throw std::runtime_error("safetensors: malformed array");
        }
        expect(']');
        return result;
    }

    void skip_value() {
        whitespace();
        const char first = peek();
        if (first == '"') { (void)string(); return; }
        if (first != '{' && first != '[') { (void)integer(); return; }
        const char open = first, close = first == '{' ? '}' : ']';
        int depth = 0;
        bool in_string = false, escaped = false;
        do {
            const char ch = text_[pos_++];
            if (in_string) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') in_string = false;
            } else if (ch == '"') {
                in_string = true;
            } else if (ch == open) {
                ++depth;
            } else if (ch == close) {
                --depth;
            }
        } while (pos_ < text_.size() && depth > 0);
        if (depth != 0) throw std::runtime_error("safetensors: unbalanced JSON value");
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
};

struct TensorMeta {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<float> read_tensor_values(std::ifstream& input, const TensorMeta& meta, uint64_t data_start, uint64_t data_size) {
    if (meta.end < meta.begin || meta.end > data_size) {
        throw std::runtime_error("invalid data range for " + meta.name);
    }
    const size_t byte_count = static_cast<size_t>(meta.end - meta.begin);
    size_t elements = 1;
    for (size_t dim : meta.shape) elements *= dim;
    const size_t element_bytes = meta.dtype == "BF16" ? 2 : meta.dtype == "F32" ? 4 : 0;
    if (!element_bytes || byte_count != elements * element_bytes) {
        throw std::runtime_error("invalid dtype or payload size for " + meta.name);
    }
    input.seekg(static_cast<std::streamoff>(data_start + meta.begin));
    if (!input) throw std::runtime_error("cannot seek tensor " + meta.name);
    std::vector<float> values(elements);

    if (meta.dtype == "F32") {
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(byte_count));
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(elements);
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
        if (!input) throw std::runtime_error("cannot read tensor " + meta.name);
        for (size_t i = 0; i < elements; ++i) {
            values[i] = bf16_to_float(raw[i]);
        }
    }
    if (!input) throw std::runtime_error("cannot read tensor " + meta.name);
    return values;
}

std::unordered_map<std::string, Tensor> load_single_safetensors_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);

    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) throw std::runtime_error("safetensors: failed to read header size");
    std::string header(static_cast<size_t>(header_size), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) throw std::runtime_error("safetensors: failed to read header");

    JsonCursor json(header);
    json.expect('{');

    std::vector<TensorMeta> metas;
    while (json.peek() != '}') {
        TensorMeta meta;
        meta.name = json.string();
        json.expect(':');
        if (meta.name == "__metadata__") {
            json.skip_value();
        } else {
            json.expect('{');
            while (json.peek() != '}') {
                const std::string key = json.string();
                json.expect(':');
                if (key == "dtype") meta.dtype = json.string();
                else if (key == "shape") meta.shape = json.array();
                else if (key == "data_offsets") {
                    const auto offsets = json.array();
                    if (offsets.size() != 2) throw std::runtime_error("safetensors: invalid offsets");
                    meta.begin = offsets[0];
                    meta.end = offsets[1];
                } else {
                    json.skip_value();
                }
                if (json.peek() == ',') json.expect(',');
            }
            json.expect('}');
            metas.push_back(std::move(meta));
        }
        if (json.peek() == ',') json.expect(',');
    }
    json.expect('}');

    file.seekg(0, std::ios::end);
    const uint64_t data_start = sizeof(header_size) + header_size;
    const uint64_t file_size = static_cast<uint64_t>(file.tellg());
    if (file_size < data_start) throw std::runtime_error("safetensors: truncated file " + path);
    std::unordered_map<std::string, Tensor> tensors;
    for (const TensorMeta& meta : metas) {
        if (meta.name == "lm_head.weight") continue; // tied to embed_tokens for this checkpoint
        auto values = read_tensor_values(file, meta, data_start, file_size - data_start);
        tensors.emplace(meta.name, Tensor(meta.shape, std::move(values)));
    }
    return tensors;
}

std::unordered_map<std::string, std::string> read_weight_map(const std::string& index_path) {
    std::ifstream file(index_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + index_path);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    JsonCursor cursor(text);
    cursor.expect('{');
    std::unordered_map<std::string, std::string> result;
    while (cursor.peek() != '}') {
        const std::string key = cursor.string();
        cursor.expect(':');
        if (key == "weight_map") {
            cursor.expect('{');
            while (cursor.peek() != '}') {
                const std::string tensor_name = cursor.string();
                cursor.expect(':');
                const std::string file_name = cursor.string();
                result[tensor_name] = file_name;
                if (cursor.peek() == ',') cursor.expect(',');
            }
            cursor.expect('}');
        } else {
            cursor.skip_value();
        }
        if (cursor.peek() == ',') cursor.expect(',');
    }
    cursor.expect('}');
    return result;
}

}  // namespace

GraniteConfig read_config(const std::string& config_path) {
    std::ifstream file(config_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + config_path);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    GraniteConfig config;
    const auto parse_uint = [&](const std::string& key, size_t& value) {
        const std::regex re("\"" + key + "\"\\s*:\\s*(\\d+)");
        std::smatch m;
        if (std::regex_search(text, m, re)) value = static_cast<size_t>(std::stoull(m[1].str()));
    };
    const auto parse_float = [&](const std::string& key, float& value) {
        const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
        std::smatch m;
        if (std::regex_search(text, m, re)) value = std::stof(m[1].str());
    };

    parse_uint("vocab_size", config.vocab_size);
    parse_uint("hidden_size", config.hidden_size);
    parse_uint("num_hidden_layers", config.num_hidden_layers);
    parse_uint("num_attention_heads", config.num_attention_heads);
    parse_uint("num_key_value_heads", config.num_key_value_heads);
    parse_uint("max_position_embeddings", config.max_position_embeddings);
    parse_uint("num_local_experts", config.num_local_experts);
    parse_uint("num_experts_per_tok", config.num_experts_per_tok);
    parse_uint("intermediate_size", config.intermediate_size);
    parse_float("rms_norm_eps", config.rms_norm_eps);
    parse_float("rope_theta", config.rope_theta);
    parse_float("attention_multiplier", config.attention_multiplier);
    parse_float("embedding_multiplier", config.embedding_multiplier);
    parse_float("residual_multiplier", config.residual_multiplier);
    parse_float("logits_scaling", config.logits_scaling);
    if (!config.num_attention_heads || !config.num_key_value_heads ||
        config.hidden_size % config.num_attention_heads ||
        config.num_attention_heads % config.num_key_value_heads ||
        !config.logits_scaling || !config.intermediate_size) {
        throw std::runtime_error("invalid Granite model configuration");
    }
    config.head_dim = config.hidden_size / config.num_attention_heads;
    return config;
}

GraniteWeights load_granite_weights(const std::string& model_dir, const GraniteConfig& config) {
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    std::unordered_map<std::string, Tensor> all_tensors;

    std::ifstream index_file(index_path, std::ios::binary);
    if (index_file.good()) {
        const auto weight_map = read_weight_map(index_path);
        std::unordered_map<std::string, bool> shards;
        for (const auto& [name, file_name] : weight_map) shards[file_name] = true;
        for (const auto& [file_name, unused] : shards) {
            auto shard = load_single_safetensors_file(model_dir + "/" + file_name);
            all_tensors.merge(shard);
        }
    } else {
        const std::string single_path = model_dir + "/model.safetensors";
        all_tensors = load_single_safetensors_file(single_path);
    }

    GraniteWeights weights;
    weights.config = config;
    const auto take = [&](const std::string& name) {
        auto node = all_tensors.extract(name);
        if (node.empty()) throw std::runtime_error("missing tensor " + name);
        return std::move(node.mapped());
    };
    weights.wte = take("model.embed_tokens.weight");
    weights.final_norm_w = take("model.norm.weight");
    weights.layers.resize(config.num_hidden_layers);

    for (size_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
        LayerWeights& layer = weights.layers[layer_idx];
        layer.ln1_w = take(prefix + "input_layernorm.weight");

        layer.q_proj_w = take(prefix + "self_attn.q_proj.weight");
        layer.k_proj_w = take(prefix + "self_attn.k_proj.weight");
        layer.v_proj_w = take(prefix + "self_attn.v_proj.weight");
        layer.o_proj_w = take(prefix + "self_attn.o_proj.weight");

        layer.ln2_w = take(prefix + "post_attention_layernorm.weight");

        layer.moe_router_w = take(prefix + "block_sparse_moe.router.layer.weight");
        layer.moe_input_w = take(prefix + "block_sparse_moe.input_linear.weight");
        layer.moe_output_w = take(prefix + "block_sparse_moe.output_linear.weight");
        if (layer.moe_input_w.dim(1) != config.intermediate_size * 2 ||
            layer.moe_output_w.dim(2) != config.intermediate_size) {
            throw std::runtime_error("MoE dimensions mismatch in " + prefix);
        }
    }

    return weights;
}

}  // namespace granite
