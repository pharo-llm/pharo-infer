"
I am the UFFI binding to the llama.cpp shared library (libllama).

I expose the minimum set of C entry points needed to run a GGUF model
end-to-end from inside a Pharo image: backend init, model/context
lifecycle, tokenization, decoding and sampling.

The library path can be overridden globally via
  AILlamaLibrary libraryPath: '/absolute/path/to/libllama.so'.

The bindings target the modern llama.cpp API (b4000 and later: the
'llama_model_load_from_file' / 'llama_init_from_model' / sampler-chain
family). On Linux look for libllama.so, on macOS libllama.dylib, on
Windows llama.dll.
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

	^ self ffiCall: #( void llama_backend_free ( void ) )
]

{ #category : 'backend' }
AILlamaLibrary >> llamaBackendInit [

	^ self ffiCall: #( void llama_backend_init ( void ) )
]

{ #category : 'batch' }
AILlamaLibrary >> llamaBatchGetOne: tokensBuffer nTokens: nTokens [
	"Build a convenience llama_batch wrapping a contiguous token array.
	 Returned by value: we receive an AILlamaBatch record."

	^ self ffiCall: #( AILlamaBatch llama_batch_get_one (
		void * tokensBuffer,
		int32 nTokens ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaContextDefaultParams [

	| params |
	params := AILlamaContextParams new.
	self llamaContextDefaultParamsInto: params getHandle.
	^ params
]

{ #category : 'model' }
AILlamaLibrary >> llamaContextDefaultParamsInto: aPointer [

	^ self ffiCall: #( void ai_llama_context_default_params ( void * aPointer ) )
]

{ #category : 'batch' }
AILlamaLibrary >> llamaDecode: ctxPtr batch: aBatch [
	"Run the forward pass for the given batch. Returns 0 on success."

	^ self ffiCall: #( int32 llama_decode (
		void * ctxPtr,
		AILlamaBatch aBatch ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaFree: ctxPtr [

	^ self ffiCall: #( void llama_free ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaGetEmbeddings: ctxPtr [
	"Return a raw float * into the embedding vector for the last decoded
	 sequence. Requires the context to have been created with
	 embeddings=true."

	^ self ffiCall: #( void * llama_get_embeddings ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaGetLogits: ctxPtr [
	"Return a raw float * into the logits of the last decoded position."

	^ self ffiCall: #( void * llama_get_logits ( void * ctxPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaInitFromModel: modelPtr params: paramsStruct [
	"Create a new llama_context * from a model and context params."

	^ self ffiCall: #( void * llama_init_from_model (
		void * modelPtr,
		AILlamaContextParams paramsStruct ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaKvSelfClear: ctxPtr [
	"Clear the KV cache (new conversation)."

	^ self ffiCall: #( void llama_kv_self_clear ( void * ctxPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelDefaultParams [

	| params |
	params := AILlamaModelParams new.
	self llamaModelDefaultParamsInto: params getHandle.
	^ params
]

{ #category : 'context' }
AILlamaLibrary >> llamaModelDefaultParamsInto: aPointer [

	^ self ffiCall: #( void ai_llama_model_default_params ( void * aPointer ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelFree: modelPtr [

	^ self ffiCall: #( void llama_model_free ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelGetVocab: modelPtr [
	"Return the opaque llama_vocab * for this model."

	^ self ffiCall: #( void * llama_model_get_vocab ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelLoadFromFile: pathString params: paramsStruct [
	"Load a model file and return an opaque llama_model * pointer.
	 paramsStruct must be an AILlamaModelParams instance."

	^ self ffiCall: #( void * llama_model_load_from_file (
		String pathString,
		AILlamaModelParams paramsStruct ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelNCtxTrain: modelPtr [
	"Maximum training context length supported by the model."

	^ self ffiCall: #( int32 llama_model_n_ctx_train ( void * modelPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaModelNEmbd: modelPtr [

	^ self ffiCall: #( int32 llama_model_n_embd ( void * modelPtr ) )
]

{ #category : 'context' }
AILlamaLibrary >> llamaNCtx: ctxPtr [

	^ self ffiCall: #( uint32 llama_n_ctx ( void * ctxPtr ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerAccept: samplerPtr token: aToken [

	^ self ffiCall: #( void llama_sampler_accept (
		void * samplerPtr,
		int32 aToken ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainAdd: chainPtr sampler: samplerPtr [

	^ self ffiCall: #( void llama_sampler_chain_add (
		void * chainPtr,
		void * samplerPtr ) )
]

{ #category : 'model' }
AILlamaLibrary >> llamaSamplerChainDefaultParams [

	| params |
	params := AILlamaSamplerChainParams new.
	self llamaSamplerChainDefaultParamsInto: params getHandle.
	^ params
]

{ #category : 'model' }
AILlamaLibrary >> llamaSamplerChainDefaultParamsInto: aPointer [

	^ self ffiCall: #( void ai_llama_sampler_chain_default_params ( void * aPointer ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerChainInit: paramsStruct [

	^ self ffiCall: #( void * llama_sampler_chain_init (
		AILlamaSamplerChainParams paramsStruct ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerFree: samplerPtr [

	^ self ffiCall: #( void llama_sampler_free ( void * samplerPtr ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerInitDist: seed [

	^ self ffiCall: #( void * llama_sampler_init_dist ( uint32 seed ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerInitGreedy [

	^ self ffiCall: #( void * llama_sampler_init_greedy ( void ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerInitTemp: t [

	^ self ffiCall: #( void * llama_sampler_init_temp ( float32 t ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerInitTopK: k [

	^ self ffiCall: #( void * llama_sampler_init_top_k ( int32 k ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerInitTopP: p minKeep: minKeep [

	^ self ffiCall: #( void * llama_sampler_init_top_p (
		float32 p,
		uint64 minKeep ) )
]

{ #category : 'sampling' }
AILlamaLibrary >> llamaSamplerSample: samplerPtr context: ctxPtr idx: idx [

	^ self ffiCall: #( int32 llama_sampler_sample (
		void * samplerPtr,
		void * ctxPtr,
		int32 idx ) )
]

{ #category : 'tokenization' }
AILlamaLibrary >> llamaTokenToPiece: vocabPtr token: aToken buf: aBuffer length: bufLen lstrip: lstrip special: special [
	"Detokenize a single token ID into a UTF-8 fragment in aBuffer.
	 Returns number of bytes written, or -needed if the buffer is too small."

	^ self ffiCall: #( int32 llama_token_to_piece (
		void * vocabPtr,
		int32 aToken,
		void * aBuffer,
		int32 bufLen,
		int32 lstrip,
		bool special ) )
]

{ #category : 'tokenization' }
AILlamaLibrary >> llamaTokenize: vocabPtr text: textString textLen: textLen tokens: tokensBuffer nTokensMax: nTokensMax addSpecial: addSpecial parseSpecial: parseSpecial [
	"Tokenize a UTF-8 string into tokensBuffer (Int32 *).
	 Returns number of tokens written on success, or a negative count
	 (-needed) when the buffer is too small."

	^ self ffiCall: #( int32 llama_tokenize (
		void * vocabPtr,
		String textString,
		int32 textLen,
		void * tokensBuffer,
		int32 nTokensMax,
		bool addSpecial,
		bool parseSpecial ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabBos: vocabPtr [

	^ self ffiCall: #( int32 llama_vocab_bos ( void * vocabPtr ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabEos: vocabPtr [

	^ self ffiCall: #( int32 llama_vocab_eos ( void * vocabPtr ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabIsEog: vocabPtr token: aToken [

	^ self ffiCall: #( bool llama_vocab_is_eog (
		void * vocabPtr,
		int32 aToken ) )
]

{ #category : 'vocab' }
AILlamaLibrary >> llamaVocabNTokens: vocabPtr [

	^ self ffiCall: #( int32 llama_vocab_n_tokens ( void * vocabPtr ) )
]

{ #category : 'configuration' }
AILlamaLibrary >> macLibraryName [

	^ self class libraryPath ifNil: [ 'libllama.dylib' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> unix64LibraryName [

	^ self class libraryPath ifNil: [ 'libllama.so' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> unixLibraryName [

	^ self class libraryPath ifNil: [ 'libllama.so' ]
]

{ #category : 'configuration' }
AILlamaLibrary >> win32LibraryName [

	^ self class libraryPath ifNil: [ 'llama.dll' ]
]
