# PharoInfer

[![Pharo 13 & 14](https://img.shields.io/badge/Pharo-13%20%7C%2014-2c98f0.svg)](https://pharo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pharo-llm/PharoInfer/pulls)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/pharo-llm/PharoInfer)

PharoInfer is a production-ready inference engine for Pharo Smalltalk that brings Large Language Models (LLMs) directly into the Pharo environment. It supports local inference, multiple model formats, streaming text generation, embeddings, and an OpenAI-compatible chat API, with backends such as native Pharo, Ollama, and llama.cpp.

## Highlights

- **Model management**: load, unload, and run multiple models.
- **Multiple formats**: GGUF, SafeTensors, PyTorch.
- **Chat & completions**: OpenAI-compatible chat completion API.
- **Embeddings**: semantic search and similarity.
- **Streaming**: token-by-token generation.
- **Backends**: local Pharo, Ollama, llama.cpp.

## Installation

### Stable

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'X.X.X' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```


### Development

```smalltalk
Metacello new
  githubUser: 'pharo-llm' project: 'pharo-infer' commitish: 'main' path: 'src';
  baseline: 'AIPharoInfer';
  load.
```

## Quick Start

### Text Completion

```smalltalk
model := AIModel fromFile: '/path/to/model.gguf' asFileReference.
model backend: AILocalBackend new.
AIModelManager default registerModel: model.
engine := AIInferenceEngine default.
engine complete: 'Tell me a story about' model: 'your-model-name'.
```

```smalltalk
| manager engine model |
manager := AIModelManager default.
manager currentBackend: AILlamaCppBackend new.
model := manager loadModel: (FileLocator home / 'path' / 'model.gguf') fullName.
engine := AIInferenceEngine default.
engine backend: manager currentBackend.
engine complete: 'Hello World !' model: model name.
```

### Chat Completion

```smalltalk
request := AIChatCompletionRequest
  model: 'your-model-name'
  messages: {
    AIChatMessage system: 'You are a helpful AI assistant'.
    AIChatMessage user: 'What is Smalltalk?' }.
AIChatAPI default complete: request.
```
