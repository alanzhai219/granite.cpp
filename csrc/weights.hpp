#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "tensor.hpp"

namespace granite {

struct LayerWeights {
    Tensor ln1_w;
    Tensor ln1_b;
    Tensor q_proj_w;
    Tensor q_proj_b;
    Tensor k_proj_w;
    Tensor k_proj_b;
    Tensor v_proj_w;
    Tensor v_proj_b;
    Tensor o_proj_w;
    Tensor o_proj_b;
    Tensor ln2_w;
    Tensor ln2_b;
    Tensor moe_router_w;
    Tensor moe_input_w;
    Tensor moe_output_w;
};

struct GraniteWeights {
    GraniteConfig config;
    Tensor wte;
    Tensor final_norm_w;
    std::vector<LayerWeights> layers;
};

GraniteConfig read_config(const std::string& config_path);
GraniteWeights load_granite_weights(const std::string& model_dir, const GraniteConfig& config);

}  // namespace granite
