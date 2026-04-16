# PharoInfer

A fully in-image inference engine for Pharo Smalltalk. PharoInfer loads
a GGUF model file directly from disk and drives
[llama.cpp](https://github.com/ggml-org/llama.cpp) through UFFI. There
is no HTTP server, no Ollama bridge, no subprocess — you interact with
the model straight from the Pharo image.

## Features

- **In-image inference** — a live `libllama` runs inside the Pharo VM
  process via UFFI.
- **GGUF models** — loaded directly from disk, mmap-backed when the
  library is compiled with mmap support.
- **Text generation** — synchronous and streaming, with a block called
  per detokenized UTF-8 piece.
- **Sampling controls** — temperature, top-k, top-p, deterministic
  seed, and a greedy fallback for `temperature = 0`.
- **Embeddings** — produced through the same loaded context.
- **GPU offload** — `nGpuLayers:` is forwarded to llama.cpp; works
  when `libllama` is built with CUDA, ROCm, Metal, or Vulkan.
- **Chat API** — a thin wrapper that formats messages and delegates to
  the inference engine.

## Requirements

- Pharo 13 or 14 (UFFI support required).
- A shared build of llama.cpp that exposes the b4000+ API:
  - Linux: `libllama.so`
  - macOS: `libllama.dylib`
  - Windows: `llama.dll`
- A `.gguf` model file.

## Installation

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'main' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```

## Configuration

### Point PharoInfer at your libllama

If the library is not on the system's default search path, pin it
explicitly:

```smalltalk
AILlamaLibrary libraryPath: '/home/me/llama.cpp/build/libllama.so'.
```

### Models directory

```smalltalk
AIInferenceConfig default
    modelsDirectory: '/path/to/models' asFileReference.
```

## Quick Start

### Basic text generation

```smalltalk
| manager engine model |
manager := AIModelManager default.
manager currentBackend: AILocalBackend new.

model := manager loadModel:
    (FileLocator home / 'models' / 'tiny.gguf') fullName.

engine := AIInferenceEngine default.
engine backend: manager currentBackend.
engine complete: 'Hello from Pharo!' model: model name.
```

### Streaming

```smalltalk
engine
    stream: 'Tell me a joke about Smalltalk'
    model: model name
    onToken: [ :piece | Transcript show: piece; flush ].
```

### Chat completion

```smalltalk
| request |
request := AIChatCompletionRequest
    model: model name
    messages: {
        AIChatMessage system: 'You are a helpful AI assistant.'.
        AIChatMessage user: 'What is Smalltalk?' }.
AIChatAPI default complete: request.
```

### Embeddings

```smalltalk
| generator |
generator := AIEmbeddingsGenerator forModel: model.
generator backend: manager currentBackend.
generator embed: 'The cat sits on the mat'.
```

### Sampling options

```smalltalk
| options |
options := AIGenerationOptions new
    temperature: 0.7;
    topP: 0.9;
    topK: 40;
    maxTokens: 256;
    seed: 42;
    yourself.
engine complete: 'Once upon a time' model: model name options: options.
```

### GPU offload and threads

```smalltalk
AILocalBackend new
    nGpuLayers: 999;  "offload every layer"
    nThreads: 8;
    contextSize: 4096.
```

## Architecture

- `AILlamaLibrary` — `FFILibrary` mapping the llama.cpp C entry points
  we call (`llama_backend_init`, `llama_model_load_from_file`,
  `llama_init_from_model`, `llama_tokenize`, `llama_decode`,
  `llama_sampler_*`, `llama_token_to_piece`, etc.).
- `AILlamaModelParams`, `AILlamaContextParams`,
  `AILlamaSamplerChainParams`, `AILlamaBatch` —
  `FFIExternalStructure` mirrors of the by-value records used by the
  C API.
- `AILocalBackend` — single production backend. Owns the model/context
  lifecycle, the tokenization path, the sampler chain, and the
  decode/sample/detokenize loop used for both synchronous and
  streaming generation.
- `AILocalModelHandle` — holds the opaque `(model *, context *)` pair
  returned by llama.cpp and releases them on unload.
- `AIGGUFParser` — optional GGUF header/metadata reader (pure Pharo).
  Useful for introspecting a model file without loading it.
- `AIInferenceEngine`, `AIChatAPI`, `AIChatMessage`,
  `AIChatCompletionRequest`, `AIChatCompletionResponse` — high-level
  API.
- `AIEmbeddingsGenerator` — wraps `generateEmbeddings:` with
  normalization and cosine-similarity helpers.
- `AIModel`, `AIModelManager`, `AIModelFormat`,
  `AIGenerationOptions`, `AIInferenceConfig` — registration, metadata
  and tuning.

## Model formats

Only GGUF is supported — that is the input format of llama.cpp.
`AIModelFormat detectFromFile:` recognises the extension and the
backend refuses anything else.

## Testing

Tests that exercise the FFI path require `libllama` to be available.
Pure logic tests (models, formats, options, manager, config) run
against `AIMockBackend` and do not need the native library.

## License

MIT License.

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the C/C++
  inference runtime PharoInfer binds to.
- The Pharo community for UFFI.
