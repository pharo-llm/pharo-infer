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
- A native PharoInfer shim linked to a shared llama.cpp build:
  - Linux: `libai_llama.so`
  - macOS: `libai_llama.dylib`
  - Windows: `ai_llama.dll`
- A `.gguf` model file.

The GGUF file is only the model. The native runtime still has to be
installed on the client machine or bundled with the image/application.

## Installation

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'main' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```

## Configuration

### Build the native runtime

On macOS, Linux, or Windows:

```sh
sh scripts/build-native.sh
```

This builds llama.cpp as a shared library, builds the PharoInfer shim,
and places the result under `$HOME/pharo-infer-native/lib`. On Windows,
run the script from Git Bash or another POSIX-compatible shell with
CMake and a C/C++ compiler available.

### Point PharoInfer at your shim

PharoInfer checks the system's default search path, the image-local
`pharo-infer-native/lib` directory, and `$HOME/pharo-infer-native/lib`.
If needed, pin the shim explicitly:

```smalltalk
AILlamaLibrary libraryPath: '/home/me/pharo-infer-native/lib/libai_llama.so'.
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

- `AILlamaLibrary` — `FFILibrary` mapping the stable PharoInfer shim
  entry points. The shim calls llama.cpp and owns the volatile C struct
  layouts.
- `AILlamaModelParams`, `AILlamaContextParams`,
  `AILlamaSamplerChainParams`, `AILlamaBatch` —
  legacy mirrors kept for source compatibility; the production backend
  no longer passes those structs through UFFI.
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

Tests that exercise the FFI path require the native shim and linked
llama.cpp library to be available.
Pure logic tests (models, formats, options, manager, config) run
against `AIMockBackend` and do not need the native library.

## License

MIT License.

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the C/C++
  inference runtime PharoInfer binds to.
- The Pharo community for UFFI.
