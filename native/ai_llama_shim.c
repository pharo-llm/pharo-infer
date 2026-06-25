#include "llama.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#  include <windows.h>
#  define AI_EXPORT __declspec(dllexport)
#else
#  include <dlfcn.h>
#  define AI_EXPORT __attribute__((visibility("default")))
#endif

AI_EXPORT struct llama_model *
ai_llama_model_load_from_file(const char * path_model, int32_t n_gpu_layers, bool use_mmap) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap = use_mmap;
    return llama_model_load_from_file(path_model, params);
}

AI_EXPORT struct llama_context *
ai_llama_init_from_model(struct llama_model * model, uint32_t n_ctx, uint32_t n_batch, int32_t n_threads, bool embeddings) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.embeddings = embeddings;
    return llama_init_from_model(model, params);
}

AI_EXPORT int32_t
ai_llama_decode_tokens(struct llama_context * ctx, int32_t * tokens, int32_t n_tokens) {
    struct llama_batch batch = llama_batch_get_one((llama_token *) tokens, n_tokens);
    return llama_decode(ctx, batch);
}

AI_EXPORT void
ai_llama_memory_clear(struct llama_context * ctx) {
    if (ctx == NULL) {
        return;
    }

#if defined(_WIN32)
    HMODULE llama = GetModuleHandleA("llama.dll");
    if (llama != NULL) {
        typedef void * (*get_memory_fn)(const struct llama_context *);
        typedef void (*memory_clear_fn)(void *, bool);
        typedef void (*kv_self_clear_fn)(struct llama_context *);

        get_memory_fn get_memory = (get_memory_fn) GetProcAddress(llama, "llama_get_memory");
        memory_clear_fn memory_clear = (memory_clear_fn) GetProcAddress(llama, "llama_memory_clear");
        if (get_memory != NULL && memory_clear != NULL) {
            memory_clear(get_memory(ctx), true);
            return;
        }

        kv_self_clear_fn kv_self_clear = (kv_self_clear_fn) GetProcAddress(llama, "llama_kv_self_clear");
        if (kv_self_clear != NULL) {
            kv_self_clear(ctx);
        }
    }
#else
    typedef void * (*get_memory_fn)(const struct llama_context *);
    typedef void (*memory_clear_fn)(void *, bool);
    typedef void (*kv_self_clear_fn)(struct llama_context *);

    get_memory_fn get_memory = (get_memory_fn) dlsym(RTLD_DEFAULT, "llama_get_memory");
    memory_clear_fn memory_clear = (memory_clear_fn) dlsym(RTLD_DEFAULT, "llama_memory_clear");
    if (get_memory != NULL && memory_clear != NULL) {
        memory_clear(get_memory(ctx), true);
        return;
    }

    kv_self_clear_fn kv_self_clear = (kv_self_clear_fn) dlsym(RTLD_DEFAULT, "llama_kv_self_clear");
    if (kv_self_clear != NULL) {
        kv_self_clear(ctx);
    }
#endif
}

AI_EXPORT int32_t
ai_llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len, int32_t * tokens, int32_t n_tokens_max, bool add_special, bool parse_special) {
    return llama_tokenize(vocab, text, text_len, (llama_token *) tokens, n_tokens_max, add_special, parse_special);
}

AI_EXPORT int32_t
ai_llama_token_to_piece(const struct llama_vocab * vocab, int32_t token, char * buf, int32_t length, int32_t lstrip, bool special) {
    return llama_token_to_piece(vocab, (llama_token) token, buf, length, lstrip, special);
}

AI_EXPORT struct llama_sampler *
ai_llama_sampler_chain_init(void) {
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    return llama_sampler_chain_init(params);
}

AI_EXPORT void
ai_llama_sampler_chain_add_top_k(struct llama_sampler * chain, int32_t k) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
}

AI_EXPORT void
ai_llama_sampler_chain_add_top_p(struct llama_sampler * chain, float p, size_t min_keep) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(p, min_keep));
}

AI_EXPORT void
ai_llama_sampler_chain_add_temp(struct llama_sampler * chain, float temperature) {
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
}

AI_EXPORT void
ai_llama_sampler_chain_add_dist(struct llama_sampler * chain, uint32_t seed) {
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
}

AI_EXPORT void
ai_llama_sampler_chain_add_greedy(struct llama_sampler * chain) {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
}

AI_EXPORT int32_t
ai_llama_sampler_sample(struct llama_sampler * sampler, struct llama_context * ctx, int32_t idx) {
    return (int32_t) llama_sampler_sample(sampler, ctx, idx);
}

AI_EXPORT void
ai_llama_sampler_free(struct llama_sampler * sampler) {
    llama_sampler_free(sampler);
}

AI_EXPORT const struct llama_vocab *
ai_llama_model_get_vocab(const struct llama_model * model) {
    return llama_model_get_vocab(model);
}

AI_EXPORT int32_t
ai_llama_model_n_embd(const struct llama_model * model) {
    return llama_model_n_embd(model);
}

AI_EXPORT int32_t
ai_llama_model_n_ctx_train(const struct llama_model * model) {
    return llama_model_n_ctx_train(model);
}

AI_EXPORT uint32_t
ai_llama_n_ctx(const struct llama_context * ctx) {
    return llama_n_ctx(ctx);
}

AI_EXPORT float *
ai_llama_get_embeddings(struct llama_context * ctx) {
    return llama_get_embeddings(ctx);
}

AI_EXPORT float *
ai_llama_get_logits(struct llama_context * ctx) {
    return llama_get_logits(ctx);
}

AI_EXPORT bool
ai_llama_vocab_is_eog(const struct llama_vocab * vocab, int32_t token) {
    return llama_vocab_is_eog(vocab, (llama_token) token);
}

AI_EXPORT int32_t
ai_llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    return llama_vocab_n_tokens(vocab);
}

AI_EXPORT int32_t
ai_llama_vocab_bos(const struct llama_vocab * vocab) {
    return (int32_t) llama_vocab_bos(vocab);
}

AI_EXPORT int32_t
ai_llama_vocab_eos(const struct llama_vocab * vocab) {
    return (int32_t) llama_vocab_eos(vocab);
}

AI_EXPORT void
ai_llama_free(struct llama_context * ctx) {
    llama_free(ctx);
}

AI_EXPORT void
ai_llama_model_free(struct llama_model * model) {
    llama_model_free(model);
}

AI_EXPORT void
ai_llama_backend_init(void) {
    llama_backend_init();
}

AI_EXPORT void
ai_llama_backend_free(void) {
    llama_backend_free();
}
