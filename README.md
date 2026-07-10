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

## Installation

Load the package in Pharo. If you are using this local checkout:

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'main' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```

## Run a GGUF model from Pharo

Do these steps in order. The `.gguf` file alone is not enough; the
client machine must also have the native llama.cpp runtime built or
bundled.

### 1. Build the native runtime

In a terminal, from this repository:

```sh
cd /path/to/pharo-infer
sh scripts/build-native.sh
```

The script clones/builds llama.cpp as a shared library and compiles the
small PharoInfer shim into `$HOME/pharo-infer-native/lib`. It supports
macOS, Linux, and Windows. On Windows, run it from Git Bash or another
POSIX-compatible shell with CMake and a C/C++ compiler available.

On macOS this creates:

```text
$HOME/pharo-infer-native/lib/libai_llama.dylib
```

On Linux this creates:

```text
$HOME/pharo-infer-native/lib/libai_llama.so
```

On Windows this creates:

```text
$HOME/pharo-infer-native/lib/ai_llama.dll
```

Keep the whole `$HOME/pharo-infer-native/lib` folder together. It
contains `libai_llama` plus the llama.cpp / ggml libraries it depends
on.

### 2. Put a model on disk

Use a GGUF model file:

```sh
mkdir -p "$HOME/pharo-models"
cp /path/to/model.gguf "$HOME/pharo-models/model.gguf"
```

### 3. Point Pharo at the native library

Pharo will look for `libai_llama.so`, `libai_llama.dylib`, or
`ai_llama.dll` on the default library search path, and also under
`FileLocator imageDirectory / 'pharo-infer-native' / 'lib'` and
`FileLocator home / 'pharo-infer-native' / 'lib'`. To override, pin it
from the image:

macOS:

```smalltalk
AILlamaLibrary libraryPath:
  (FileLocator home / 'pharo-infer-native' / 'lib' / 'libai_llama.dylib') fullName.
```

Linux:

```smalltalk
AILlamaLibrary libraryPath:
  (FileLocator home / 'pharo-infer-native' / 'lib' / 'libai_llama.so') fullName.
```

Windows:

```smalltalk
AILlamaLibrary libraryPath:
  (FileLocator home / 'pharo-infer-native' / 'lib' / 'ai_llama.dll') fullName.
```

### 4. Load the model and ask it something

Run this in a Pharo Playground:

```smalltalk
| backend manager engine model answer |

backend := AILocalBackend new
  nThreads: 8;
  contextSize: 2048;
  batchSize: 512;
  nGpuLayers: 0;
  yourself.

manager := AIModelManager new.
manager currentBackend: backend.

model := manager loadModel:
  (FileLocator home / 'pharo-models' / 'model.gguf') fullName.

engine := AIInferenceEngine new.
engine modelManager: manager.

answer := engine
  complete: 'Say hello from Pharo in one short sentence.'
  model: model name.

Transcript show: answer; cr.
answer
```

For Apple Silicon / Metal GPU offload, try this before loading the
model:

```smalltalk
backend nGpuLayers: 999.
```

## More examples

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
