# PharoInfer

A comprehensive, production-ready inference engine for Pharo Smalltalk that brings the power of Large Language Models (LLMs) directly into your Pharo environment. PharoInfer supports all the features you'd expect from modern inference engines like Ollama and llama.cpp, with native Pharo integration.

## Features

### Core Capabilities

- **Model Management**: Load, unload, and manage multiple LLM models
- **Multiple Format Support**: GGUF, SafeTensors, PyTorch formats
- **Text Generation**: Synchronous and streaming text completion
- **Embeddings**: Generate vector embeddings for semantic search and similarity
- **Chat API**: OpenAI-compatible chat completion interface
- **Tokenization**: Built-in text tokenization and processing

### Backend Support

- **Local Backend**: Native Pharo inference engine
- **Ollama Integration**: Connect to Ollama servers
- **llama.cpp Integration**: Interface with llama.cpp server instances

### Advanced Features

- **Streaming**: Real-time token-by-token generation
- **Generation Options**: Temperature, top-p, top-k, max tokens, penalties
- **GPU Acceleration**: Configurable GPU support (when backend supports it)
- **Context Management**: Configurable context windows
- **Multiple Models**: Run multiple models concurrently

## Installation

### From Iceberg/Git

```smalltalk
Metacello new
  baseline: 'AIPharoInfer';
  repository: 'github://yourusername/PharoInfer:main/src';
  load.
```

### Manual Installation

1. Clone this repository
2. Open Pharo
3. Load the packages via Iceberg or Monticello

## Quick Start

### Basic Text Generation

```smalltalk
"Create and register a model"
model := AIModel fromFile: '/path/to/model.gguf' asFileReference.
model backend: AILocalBackend new.
manager := AIModelManager default.
manager registerModel: model.
"Generate text"
engine := AIInferenceEngine default.
result := engine complete: 'Tell me a story about' model: 'your-model-name'.
```

### Chat Completion

```smalltalk
"Create a chat request"
request := AIChatCompletionRequest
  model: 'your-model-name'
  messages: {
    AIChatMessage system: 'You are a helpful AI assistant'.
    AIChatMessage user: 'What is Smalltalk?' }.
"Get completion"
api := AIChatAPI default.
response := api complete: request.
"Access the response"
response message content. "=> 'Smalltalk is...'"
```

### Streaming Generation

```smalltalk
"Stream text generation"
engine := AIInferenceEngine default.
engine
  stream: 'Once upon a time'
  model: 'your-model-name'
  onToken: [ :token |
    Transcript show: token; flush ].
```

### Generate Embeddings

```smalltalk
"Create embeddings generator"
generator := AIEmbeddingsGenerator forModel: yourModel.
"Generate embeddings"
embedding1 := generator embed: 'The cat sits on the mat'.
embedding2 := generator embed: 'A feline rests on the rug'.
"Compute similarity"
similarity := generator cosineSimilarity: embedding1 with: embedding2.
```

## Configuration

### Setting Up Models Directory

```smalltalk
config := AIInferenceConfig default.
config modelsDirectory: '/path/to/models' asFileReference.
```

### Configuring Generation Options

```smalltalk
options := AIGenerationOptions new
  temperature: 0.7;
  maxTokens: 500;
  topP: 0.9;
  topK: 40;
  repeatPenalty: 1.1;
  yourself.
result := engine complete: 'Your prompt' model: 'model-name' options: options.
```

### GPU Configuration

```smalltalk
config := AIInferenceConfig default.
config enableGPU: true.
config gpuLayers: 32.  "Number of layers to offload to GPU"
```

## Backend Integration

### Using Ollama Backend

```smalltalk
"Configure Ollama backend"
ollamaBackend := AIOllamaBackend new.
ollamaBackend baseUrl: 'http://localhost:11434'.
"Set as model backend"
model backend: ollamaBackend.
"Use as normal"
result := engine complete: 'Hello' model: 'llama2'.
```

### Using llama.cpp Backend

```smalltalk
"Configure llama.cpp backend"
llamaCppBackend := AILlamaCppBackend new.
llamaCppBackend serverUrl: 'http://localhost:8080'.
"Set as model backend"
model backend: llamaCppBackend.
"Use as normal"
result := engine complete: 'Hello' model: 'your-model'.
```

## Architecture

PharoInfer is built with a clean, extensible architecture:

### Core Components

- **AIModel**: Represents a loaded or loadable LLM model
- **AIModelManager**: Manages model lifecycle (loading, unloading, discovery)
- **AIInferenceEngine**: Main inference engine for text generation
- **AIBackend**: Abstract backend interface for different inference providers

### API Layer

- **AIChatAPI**: OpenAI-compatible chat completion API
- **AIChatMessage**: Represents chat messages (system, user, assistant)
- **AIGenerationOptions**: Configuration for text generation

### Tokenization

- **AITokenizer**: Text tokenization and encoding/decoding

### Embeddings

- **AIEmbeddingsGenerator**: Generate and manipulate vector embeddings

### Configuration

- **AIInferenceConfig**: Global configuration for the inference engine
- **AIModelFormat**: Model format detection and handling

## Model Formats

PharoInfer supports multiple model formats:

### GGUF (GPT-Generated Unified Format)

The recommended format for llama.cpp and Ollama models. Optimized for:
- Fast loading with memory mapping
- Various quantization levels (2-bit to 8-bit)
- Efficient CPU inference

### SafeTensors

A safe, fast serialization format for ML models:
- Zero-copy loading
- Safe from arbitrary code execution
- Wide framework support

### PyTorch

Standard PyTorch model format:
- `.pt` and `.pth` files
- Requires PyTorch-compatible backend

## Testing

PharoInfer includes comprehensive test coverage:

### Running Tests

```smalltalk
"Run all tests"
AIModelTest suite run.
AIModelManagerTest suite run.
AITokenizerTest suite run.
AIInferenceEngineTest suite run.
AIChatAPITest suite run.
AIEmbeddingsGeneratorTest suite run.
AIIntegrationTest suite run.
```

### Test Coverage

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows end-to-end
- **Backend Tests**: Verify backend integrations work correctly

## Examples

### Complete Workflow Example

```smalltalk
"1. Configure the system"
config := AIInferenceConfig default.
config modelsDirectory: '/path/to/models' asFileReference.
"2. Discover models in directory"
manager := AIModelManager default.
discovered := manager discoverModelsInDirectory: config modelsDirectory.
"3. Load a specific model"
model := manager loadModel: 'llama-2-7b.gguf'.
"4. Configure generation options"
options := AIGenerationOptions new
  temperature: 0.8;
  maxTokens: 200;
  yourself.
"5. Generate text"
engine := AIInferenceEngine default.
result := engine complete: 'Explain quantum computing' model: 'llama-2-7b' options: options.
"6. Clean up"
manager unloadModel: 'llama-2-7b'.
```

### Semantic Search with Embeddings

```smalltalk
"1. Setup"
model := manager loadModel: 'embeddings-model'.
generator := AIEmbeddingsGenerator forModel: model.
"2. Create document embeddings"
documents := {
  'Pharo is a pure object-oriented programming language'.
  'Python is widely used for data science'.
  'Smalltalk inspired many modern languages' }.
embeddings := generator embedBatch: documents.
"3. Query"
query := 'Tell me about Pharo'.
queryEmbedding := generator embed: query.
"4. Find most similar"
similarities := embeddings collect: [ :docEmb |
  generator cosineSimilarity: queryEmbedding with: docEmb ].
"5. Get best match"
bestIdx := similarities indexOf: similarities max.
bestMatch := documents at: bestIdx.
```

### Multi-Turn Conversation

```smalltalk
"Setup"
api := AIChatAPI default.
conversation := OrderedCollection new.
"Add system prompt"
conversation add: (AIChatMessage system: 'You are a Pharo programming expert').
"First turn"
conversation add: (AIChatMessage user: 'What is a block in Pharo?').
request := AIChatCompletionRequest model: 'your-model' messages: conversation asArray.
response := api complete: request.
conversation add: response message.
"Second turn"
conversation add: (AIChatMessage user: 'Can you show me an example?').
request := AIChatCompletionRequest model: 'your-model' messages: conversation asArray.
response := api complete: request.
conversation add: response message.
"Access full conversation"
conversation do: [ :msg |
  Transcript show: msg role; show: ': '; show: msg content; cr ].
```

## Performance Considerations

### Memory Management

- Models are loaded on-demand and can be unloaded to free memory
- Use `AIModelManager unloadAll` to free all loaded models
- Configure `maxConcurrentInferences` based on available memory

### Quantization

For better performance with limited resources:
- Use GGUF models with appropriate quantization (4-bit or 8-bit)
- Lower quantization = less memory, faster inference, slightly lower quality

### Batch Processing

When processing multiple requests:
- Use embeddings batch processing: `embedBatch:`
- Configure `batchSize` in `AIInferenceConfig`

## Troubleshooting

### Model Loading Issues

```smalltalk
"Check if model file exists"
model path exists. "=> should be true"
"Check model format"
model format. "=> should match file extension"
"Verify backend is set"
model backend. "=> should not be nil"
```

### Memory Issues

```smalltalk
"Unload unused models"
AIModelManager default unloadAll.
"Check loaded models"
AIModelManager default allModels select: #isLoaded.
"Reduce context size"
config contextSize: 1024.  "Instead of 2048"
```

## Contributing

Contributions are welcome! Areas for improvement:

1. Additional model format support
2. More sophisticated tokenization (BPE, WordPiece)
3. Model quantization tools
4. Performance optimizations
5. Additional backend integrations (vLLM, TensorRT-LLM)
6. Web interface for model management

## License

MIT License

## Acknowledgments

Built with inspiration from:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Efficient LLM inference in C/C++
- [Ollama](https://ollama.ai) - Easy local LLM deployment
- The Pharo community for their excellent development environment

## References

This inference engine implements features and concepts from modern LLM inference systems:

- [Llama.cpp vs Ollama Comparison](https://www.openxcell.com/blog/llama-cpp-vs-ollama/)
- [Local LLM Deployment Guide](https://www.oreateai.com/blog/ollama-vs-llamacpp-navigating-the-landscape-of-local-llm-deployment/c041c18ab9d4ad1e1735b470ce07fcf5)
- [vLLM vs llama.cpp](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)

## Version

**1.0.0** - Initial release with full inference engine capabilities
