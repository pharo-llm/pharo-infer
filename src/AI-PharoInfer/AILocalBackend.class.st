"
I am a fully in-image inference backend driven by llama.cpp via UFFI.

I read a GGUF file directly from disk through libllama (no external
server, no subprocess), then let the Pharo image issue tokenization,
forward-pass and sampling calls against the loaded model.

```smalltalk
| manager engine model |
manager := AIModelManager default.
manager currentBackend: AILocalBackend new.
model := manager loadModel: (FileLocator home / 'models' / 'tiny.gguf') fullName.
engine := AIInferenceEngine default.
engine backend: manager currentBackend.
engine complete: 'Hello World !' model: model name.
```

Native shim location can be overridden globally via
  AILlamaLibrary libraryPath: '/abs/path/to/libai_llama.so'.
"
Class {
	#name : 'AILocalBackend',
	#superclass : 'AIBackend',
	#instVars : [
		'library',
		'loadedModels',
		'nThreads',
		'contextSize',
		'batchSize',
		'nGpuLayers',
		'useMmap',
		'backendInitialized',
		'embeddings'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'private' }
AILocalBackend >> buildSampler: options [

	| lib chain seed |
	lib := self library.
	chain := lib llamaSamplerChainInit.

	options topK > 0 ifTrue: [
		lib llamaSamplerChainAddTopK: chain k: options topK ].

	options topP < 1.0 ifTrue: [
		lib
			llamaSamplerChainAddTopP: chain
			p: options topP
			minKeep: 1 ].

	options temperature > 0.0
		ifTrue: [
			lib llamaSamplerChainAddTemp: chain temperature: options temperature.
			seed := options seed ifNil: [ 16rFFFFFFFF ].
			lib llamaSamplerChainAddDist: chain seed: seed ]
		ifFalse: [
			lib llamaSamplerChainAddGreedy: chain ].

	^ chain
]

{ #category : 'accessing' }
AILocalBackend >> batchSize [

	^ batchSize ifNil: [ 512 ]
]

{ #category : 'accessing' }
AILocalBackend >> batchSize: anInteger [

	batchSize := anInteger
]

{ #category : 'accessing' }
AILocalBackend >> contextSize [

	^ contextSize ifNil: [ 2048 ]
]

{ #category : 'accessing' }
AILocalBackend >> contextSize: anInteger [

	contextSize := anInteger
]

{ #category : 'private' }
AILocalBackend >> decodeTokens: tokenArray context: ctxPtr [

	| lib buffer rc |
	lib := self library.
	buffer := FFIExternalArray
		externalNewType: 'int32'
		size: tokenArray size.
	[
		1 to: tokenArray size do: [ :i |
			buffer at: i put: (tokenArray at: i) ].
		rc := lib
			llamaDecodeTokens: ctxPtr
			tokens: buffer getHandle
			nTokens: tokenArray size.
		rc ~= 0 ifTrue: [
			Error signal: 'llama_decode failed with code ', rc asString ] ]
		ensure: [ buffer free ]
]

{ #category : 'private' }
AILocalBackend >> detokenize: tokenId vocab: vocabPtr [

	| lib buf written result |
	lib := self library.
	buf := ByteArray new: 64.
	written := lib
		llamaTokenToPiece: vocabPtr
		token: tokenId
		buf: buf
		length: buf size
		lstrip: 0
		special: false.
	written < 0 ifTrue: [
		buf := ByteArray new: written negated.
		written := lib
			llamaTokenToPiece: vocabPtr
			token: tokenId
			buf: buf
			length: buf size
			lstrip: 0
			special: false ].
	written <= 0 ifTrue: [ ^ '' ].
	result := buf copyFrom: 1 to: written.
	^ [ result utf8Decoded ]
		on: Error
		do: [ :e | result asString ]
]

{ #category : 'accessing' }
AILocalBackend >> embeddings [

	^ embeddings ifNil: [ false ]
]

{ #category : 'accessing' }
AILocalBackend >> embeddings: aBoolean [

	embeddings := aBoolean
]

{ #category : 'private' }
AILocalBackend >> ensureBackendInitialized [

	backendInitialized ifTrue: [ ^ self ].
	self library llamaBackendInit.
	backendInitialized := true
]

{ #category : 'private' }
AILocalBackend >> ensureModelLoaded: aModel [

	(loadedModels includesKey: aModel name) ifFalse: [
		Error signal: 'Model not loaded: ', aModel name ]
]

{ #category : 'operations' }
AILocalBackend >> generateEmbeddings: text model: aModel [
	"Compute a pooled embedding vector for text using the loaded context.
	 The context must have been created with embeddings=true. Build the
	 backend with #embeddings: true before calling #loadModel: if you
	 intend to use this method."

	| handle vocabPtr tokens embPtr embd dim |
	self ensureModelLoaded: aModel.
	handle := loadedModels at: aModel name.
	vocabPtr := self library llamaModelGetVocab: handle modelPtr.

	tokens := self tokenizeString: text vocab: vocabPtr addSpecial: true.
	
	self library llamaKvSelfClear: handle ctxPtr.

	self decodeTokens: tokens context: handle ctxPtr.

	dim := self library llamaModelNEmbd: handle modelPtr.
	embPtr := self library llamaGetEmbeddings: handle ctxPtr.
	embPtr isNull ifTrue: [
		Error signal: 'llama_get_embeddings returned NULL; create the backend with #embeddings: true to enable pooled embeddings.' ].

	embd := Array new: dim.
	1 to: dim do: [ :i | embd at: i put: (embPtr float32AtOffset: i - 1 * 4) ].
	^ embd
]

{ #category : 'operations' }
AILocalBackend >> generateText: prompt model: aModel options: options [

	| out |
	out := WriteStream on: String new.
	self
		streamText: prompt
		model: aModel
		options: options
		onToken: [ :piece | out nextPutAll: piece ].
	^ out contents
]

{ #category : 'initialization' }
AILocalBackend >> initialize [

	super initialize.
	loadedModels := Dictionary new.
	backendInitialized := false.
	embeddings := false
]

{ #category : 'private' }
AILocalBackend >> library [

	^ library ifNil: [ library := AILlamaLibrary uniqueInstance ]
]

{ #category : 'operations' }
AILocalBackend >> loadModel: aModel [

	| modelPtr ctxPtr handle path |
	(loadedModels includesKey: aModel name) ifTrue: [ ^ self ].

	aModel path ifNil: [ Error signal: 'Model has no path: ' , aModel name ].
	aModel path exists ifFalse: [ Error signal: 'Model file not found: ' , aModel path fullName ].
	aModel format ifNil: [
		aModel format: (AIModelFormat detectFromFile: aModel path) ].
	aModel format extension = 'gguf' ifFalse: [
		Error signal: 'Only GGUF models can be loaded by AILocalBackend: ' , aModel path fullName ].

	self ensureBackendInitialized.

	path := aModel path fullName.

	modelPtr := self library
		llamaModelLoadFromFile: path
		nGpuLayers: self nGpuLayers
		useMmap: self useMmap.
	modelPtr isNull ifTrue: [ Error signal: 'llama_model_load_from_file failed for: ' , path ].

	ctxPtr := self library
		llamaInitFromModel: modelPtr
		contextSize: self contextSize
		batchSize: self batchSize
		threads: self nThreads
		embeddings: self embeddings.
	ctxPtr isNull ifTrue: [
			self library llamaModelFree: modelPtr.
			Error signal: 'llama_init_from_model failed for: ' , path ].

	handle := AILocalModelHandle new
		          modelPtr: modelPtr;
		          ctxPtr: ctxPtr;
		          path: path;
		          yourself.

	loadedModels at: aModel name put: handle
]

{ #category : 'accessing' }
AILocalBackend >> nGpuLayers [

	^ nGpuLayers ifNil: [ 0 ]
]

{ #category : 'accessing' }
AILocalBackend >> nGpuLayers: anInteger [

	nGpuLayers := anInteger
]

{ #category : 'accessing' }
AILocalBackend >> nThreads [

	^ nThreads ifNil: [ 4 ]
]

{ #category : 'accessing' }
AILocalBackend >> nThreads: anInteger [

	nThreads := anInteger
]

{ #category : 'accessing' }
AILocalBackend >> useMmap [

	^ useMmap ifNil: [ true ]
]

{ #category : 'accessing' }
AILocalBackend >> useMmap: aBoolean [

	useMmap := aBoolean
]

{ #category : 'operations' }
AILocalBackend >> streamText: prompt model: aModel options: options onToken: aBlock [

	| handle vocabPtr lib tokens sampler generated maxTokens nextToken piece |
	self ensureModelLoaded: aModel.
	handle := loadedModels at: aModel name.
	lib := self library.

	vocabPtr := lib llamaModelGetVocab: handle modelPtr.
	tokens := self tokenizeString: prompt vocab: vocabPtr addSpecial: true.
	lib llamaKvSelfClear: handle ctxPtr.
	self decodeTokens: tokens context: handle ctxPtr.

	sampler := self buildSampler: options.
	maxTokens := options maxTokens.
	generated := 0.

	[
		[ generated < maxTokens ] whileTrue: [
				nextToken := lib llamaSamplerSample: sampler context: handle ctxPtr idx: -1.
				(lib llamaVocabIsEog: vocabPtr token: nextToken) ifTrue: [ ^ self ]. "llama_sampler_sample already accepts the chosen token; do not
			 double-accept or stateful samplers (penalties, mirostat) drift."
				piece := self detokenize: nextToken vocab: vocabPtr.
				piece ifNotEmpty: [ aBlock value: piece ].

				self decodeTokens: (Array with: nextToken) context: handle ctxPtr.
				generated := generated + 1 ] ] ensure: [ lib llamaSamplerFree: sampler ]
]

{ #category : 'private' }
AILocalBackend >> tokenizeString: aString vocab: vocabPtr addSpecial: addSpecial [

	| lib bytes initialCap buffer written needed tokens |
	lib := self library.
	bytes := aString utf8Encoded.
	initialCap := bytes size + 16 max: 32.
	buffer := FFIExternalArray
		externalNewType: 'int32'
		size: initialCap.
	[
		written := lib
			llamaTokenize: vocabPtr
			textBytes: bytes
			textLen: bytes size
			tokens: buffer getHandle
			nTokensMax: initialCap
			addSpecial: addSpecial
			parseSpecial: true.
		written < 0 ifTrue: [
			needed := written negated.
			buffer free.
			buffer := FFIExternalArray
				externalNewType: 'int32'
				size: needed.
			written := lib
				llamaTokenize: vocabPtr
				textBytes: bytes
				textLen: bytes size
				tokens: buffer getHandle
				nTokensMax: needed
				addSpecial: addSpecial
				parseSpecial: true ].
		tokens := Array new: written.
		1 to: written do: [ :i | tokens at: i put: (buffer at: i) ] ]
		ensure: [ buffer free ].
	^ tokens
]

{ #category : 'operations' }
AILocalBackend >> unloadModel: aModel [

	| handle |
	handle := loadedModels removeKey: aModel name ifAbsent: [ ^ self ].
	handle freeWith: self library
]
