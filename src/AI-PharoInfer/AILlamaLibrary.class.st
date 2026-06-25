"
I am the UFFI binding to PharoInfer's small llama.cpp shim library.

Do not bind Pharo directly to llama.cpp structs here. llama.cpp changes
the layout of by-value structs such as llama_model_params,
llama_context_params and llama_batch across releases. The shim is
compiled against the user's installed llama.cpp headers and exposes a
stable pointer/int/string ABI to Pharo.

Build the shim with:
  sh scripts/build-native.sh

Then point the image at it:
  AILlamaLibrary libraryPath: '/absolute/path/to/libai_llama.dylib'.
"
Class {
	#name : 'AILlamaLibrary',
	#superclass : 'FFILibrary',
	#classInstVars : [
		'libraryPath'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AILlamaLibrary class >> libraryPath [

	^ libraryPath
]

{ #category : 'accessing' }
AILlamaLibrary class >> libraryPath: aString [

	libraryPath := aString
]

{ #category : 'accessing' }
AILlamaLibrary class >> reset [
	<script>
	libraryPath := nil
]

{ #category : 'backend' }
AILlamaLibrary >> llamaBackendFree [

	^ self ffiCall: #( void ai_llama_backend_free ( void ) )
]

{ #category : 'backend' }
AILlamaLibrary >> llamaBackendInit [

	^ self ffiCall: #( void ai_llama_backend_init ( void ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaDecodeTokens: ctxPtr tokens: tokensBuffer nTokens: nTokens [
	"Decode a contiguous int32 token array. The C shim builds llama_batch
	inside native code so Pharo does not need to mirror that struct."

	^ self
		ffiCall: #( int32 ai_llama_decode_tokens (
			void * ctxPtr,
			void * tokensBuffer,
			int32 nTokens ) )
		options: #( optThreaded )
]

{ #category : 'context' }
AILlamaLibrary >> llamaFree: ctxPtr [

	^ self ffiCall: #( void ai_llama_free ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaGetEmbeddings: ctxPtr [

	^ self ffiCall: #( void * ai_llama_get_embeddings ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaGetLogits: ctxPtr [

	^ self ffiCall: #( void * ai_llama_get_logits ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaInitFromModel: modelPtr contextSize: nCtx batchSize: nBatch threads: nThreads embeddings: embeddingsEnabled [
	"Create a llama_context * using default llama.cpp params plus the
	Pharo-controlled values. The shim owns the real struct layout."

	^ self
		ffiCall: #( void * ai_llama_init_from_model (
			void * modelPtr,
			uint32 nCtx,
			uint32 nBatch,
			int32 nThreads,
			bool embeddingsEnabled ) )
		options: #( optThreaded )
]

{ #category : 'context' }
AILlamaLibrary >> llamaKvSelfClear: ctxPtr [
	"Clear the current context memory. The shim supports both current
	llama_memory_clear() and older llama_kv_self_clear() runtimes."

	^ self ffiCall: #( void ai_llama_memory_clear ( void * ctxPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelFree: modelPtr [

	^ self ffiCall: #( void ai_llama_model_free ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelGetVocab: modelPtr [

	^ self ffiCall: #( void * ai_llama_model_get_vocab ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelLoadFromFile: pathString nGpuLayers: nGpuLayers useMmap: useMmap [

	^ self
		ffiCall: #( void * ai_llama_model_load_from_file (
			String pathString,
			int32 nGpuLayers,
			bool useMmap ) )
		options: #( optThreaded )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelNCtxTrain: modelPtr [

	^ self ffiCall: #( int32 ai_llama_model_n_ctx_train ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelNEmbd: modelPtr [

	^ self ffiCall: #( int32 ai_llama_model_n_embd ( void * modelPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaNCtx: ctxPtr [

	^ self ffiCall: #( uint32 ai_llama_n_ctx ( void * ctxPtr ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAddDist: chainPtr seed: seed [

	^ self ffiCall: #( void ai_llama_sampler_chain_add_dist (
		void * chainPtr,
		uint32 seed ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAddGreedy: chainPtr [

	^ self ffiCall: #( void ai_llama_sampler_chain_add_greedy ( void * chainPtr ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAddTemp: chainPtr temperature: temperature [

	^ self ffiCall: #( void ai_llama_sampler_chain_add_temp (
		void * chainPtr,
		float32 temperature ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAddTopK: chainPtr k: k [

	^ self ffiCall: #( void ai_llama_sampler_chain_add_top_k (
		void * chainPtr,
		int32 k ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAddTopP: chainPtr p: p minKeep: minKeep [

	^ self ffiCall: #( void ai_llama_sampler_chain_add_top_p (
		void * chainPtr,
		float32 p,
		uint64 minKeep ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainInit [

	^ self ffiCall: #( void * ai_llama_sampler_chain_init ( void ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerFree: samplerPtr [

	^ self ffiCall: #( void ai_llama_sampler_free ( void * samplerPtr ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerSample: samplerPtr context: ctxPtr idx: idx [

	^ self
		ffiCall: #( int32 ai_llama_sampler_sample (
			void * samplerPtr,
			void * ctxPtr,
			int32 idx ) )
		options: #( optThreaded )
]

{ #category : 'tokenization' }
AILlamaLibrary >> llamaTokenToPiece: vocabPtr token: aToken buf: aBuffer length: bufLen lstrip: lstrip special: special [

	^ self ffiCall: #( int32 ai_llama_token_to_piece (
		void * vocabPtr,
		int32 aToken,
		void * aBuffer,
		int32 bufLen,
		int32 lstrip,
		bool special ) )
]

{ #category : 'tokenization' }
AILlamaLibrary >> llamaTokenize: vocabPtr textBytes: textBytes textLen: textLen tokens: tokensBuffer nTokensMax: nTokensMax addSpecial: addSpecial parseSpecial: parseSpecial [

	^ self ffiCall: #( int32 ai_llama_tokenize (
		void * vocabPtr,
		void * textBytes,
		int32 textLen,
		void * tokensBuffer,
		int32 nTokensMax,
		bool addSpecial,
		bool parseSpecial ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabBos: vocabPtr [

	^ self ffiCall: #( int32 ai_llama_vocab_bos ( void * vocabPtr ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabEos: vocabPtr [

	^ self ffiCall: #( int32 ai_llama_vocab_eos ( void * vocabPtr ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabIsEog: vocabPtr token: aToken [

	^ self ffiCall: #( bool ai_llama_vocab_is_eog (
		void * vocabPtr,
		int32 aToken ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabNTokens: vocabPtr [

	^ self ffiCall: #( int32 ai_llama_vocab_n_tokens ( void * vocabPtr ) )
]

{ #category : 'configuration' }
AILlamaLibrary >> macLibraryName [

	^ self class libraryPath ifNil: [ 'libai_llama.dylib' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> unix64LibraryName [

	^ self class libraryPath ifNil: [ 'libai_llama.so' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> unixLibraryName [

	^ self class libraryPath ifNil: [ 'libai_llama.so' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> win32LibraryName [

	^ self class libraryPath ifNil: [ 'ai_llama.dll' ]
]
