#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_activation_extractor
//

// Stores extracted activations for FINTs vector generation
struct llama_activation_extractor {
    bool enabled = false;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_tokens = 0;
    
    // Storage for activations: [n_layer][n_tokens * n_embd]
    std::vector<std::vector<float>> attn_activations;  // attention outputs
    std::vector<std::vector<float>> mlp_activations;   // MLP/FFN outputs
    
    // Token count per layer (for averaging)
    std::vector<int32_t> token_counts;
    
    // Tensor references for extraction (filled during graph construction)
    std::vector<ggml_tensor*> attn_tensors;  // per layer
    std::vector<ggml_tensor*> mlp_tensors;   // per layer
    
    void init(int32_t n_layer, int32_t n_embd);
    void clear();
    void save_attn(int il, const float * data, int n_tokens, int n_embd);
    void save_mlp(int il, const float * data, int n_tokens, int n_embd);
    
    // Extract activations from stored tensor references (call after graph compute)
    void extract_from_tensors();
    
    // Compute averaged vectors: (positive - negative) / n_tokens
    static void compute_vectors(
        const llama_activation_extractor & positive,
        const llama_activation_extractor & negative,
        std::vector<float> & attn_vec_out,
        std::vector<float> & mlp_vec_out);
};

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;
    ggml_tensor * tensor_for_attn(int il) const;  // FINTs: attention control vector
    ggml_tensor * tensor_for_mlp(int il) const;   // FINTs: MLP control vector

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;
    ggml_tensor * apply_to_attn(ggml_context * ctx, ggml_tensor * cur, int  il) const;  // FINTs
    ggml_tensor * apply_to_mlp(ggml_context * ctx, ggml_tensor * cur, int  il) const;   // FINTs

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

    // FINTs: Apply fine-grained control vectors
    bool apply_fints(
            const llama_model & model,
            const float * attn_data,
            const float * mlp_data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);
    bool init_fints(const llama_model & model);  // FINTs

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer (original)
    std::vector<ggml_tensor *> tensors_attn; // per layer (FINTs: attention)
    std::vector<ggml_tensor *> tensors_mlp;  // per layer (FINTs: MLP)
    bool use_fints = false;  // Flag to indicate if using fine-grained control
};

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    llama_adapter_lora() = default;
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
