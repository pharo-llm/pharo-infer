# PharoInfer

[![Pharo 13 & 14](https://img.shields.io/badge/Pharo-13%20%7C%2014-2c98f0.svg)](https://pharo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pharo-llm/PharoInfer/pulls)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/pharo-llm/PharoInfer)

PharoInfer is a **fully in-image** inference engine for Pharo
Smalltalk. It loads a GGUF model file directly from disk and drives
[llama.cpp](https://github.com/ggml-org/llama.cpp) through UFFI — there
is no HTTP server, no Ollama bridge, and no subprocess. Talk to the
model straight from the image.

## Requirements

- Pharo 13 or 14 (UFFI must be available).
- A native PharoInfer shim (`libai_llama.so` on Linux,
  `libai_llama.dylib` on macOS, `ai_llama.dll` on Windows), linked to
  a shared build of llama.cpp.
- A `.gguf` model file.

`model.gguf` is the model weights only. The client machine also needs
the native llama.cpp runtime, either installed by the user or shipped
with your Pharo image/application.

### Build the native runtime

From this repository on macOS or Linux:

```sh
sh scripts/build-native.sh
```

The script clones/builds llama.cpp as a shared library and compiles the
small PharoInfer shim into `$HOME/pharo-infer-native/lib`.

### Point PharoInfer at the shim

Pharo will look for `libai_llama.so` / `libai_llama.dylib` on the
default library search path. To override, pin it from the image:

```smalltalk
AILlamaLibrary libraryPath: '/home/me/pharo-infer-native/lib/libai_llama.so'.
```

## Installation

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'main' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```

## Quick Start

### Text completion, in-image

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
    onToken: [ :piece | Transcript show: piece ].
```

### Chat

```smalltalk
| request |
request := AIChatCompletionRequest
    model: model name
    messages: {
        AIChatMessage system: 'You are a helpful AI assistant.'.
        AIChatMessage user: 'What is Smalltalk?' }.
AIChatAPI default complete: request.
```

### GPU offload and threads

```smalltalk
AILocalBackend new
    nGpuLayers: 999; "offload all layers"
    nThreads: 8;
    contextSize: 4096.
```

## Architecture

- `AILlamaLibrary` — `FFILibrary` mapping the PharoInfer shim entry
  points.
- `AILlamaModelParams`, `AILlamaContextParams`, `AILlamaBatch`,
  `AILlamaSamplerChainParams` — legacy structure mirrors kept for
  source compatibility; the production backend no longer passes them
  through UFFI.
- `AILocalBackend` — drives llama.cpp: loads a model, runs
  tokenization + decode + sampling, and detokenizes back to UTF-8.
- `AILocalModelHandle` — owns the opaque `(model *, context *)` pair
  and frees it on unload.
- `AIGGUFParser` — optional pre-flight reader for GGUF metadata
  (header, vocab, special tokens) without loading the model.
- `AIInferenceEngine`, `AIChatAPI` — high-level entry points.
