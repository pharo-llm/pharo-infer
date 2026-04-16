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

libllama location can be overridden globally via
  AILlamaLibrary libraryPath: '/abs/path/to/libllama.so'.
"
Class {
	#name : 'AILocalBackend',
	#superclass : 'AIBackend',
	#instVars : [
		'library',
		'loadedModels',
		'nThreads',
		'contextSize',
		'nGpuLayers',
		'backendInitialized',
		'embeddings'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'private' }
AILocalBackend >> buildSampler: options [

	| lib chainParams chain seed |
	lib := self library.
	chainParams := lib llamaSamplerChainDefaultParams.
	chain := lib llamaSamplerChainInit: chainParams.

	options topK > 0 ifTrue: [
		lib
			llamaSamplerChainAdd: chain
			sampler: (lib llamaSamplerInitTopK: options topK) ].

	options topP < 1.0 ifTrue: [
		lib
			llamaSamplerChainAdd: chain
			sampler: (lib llamaSamplerInitTopP: options topP minKeep: 1) ].

	options temperature > 0.0
		ifTrue: [
			lib
				llamaSamplerChainAdd: chain
				sampler: (lib llamaSamplerInitTemp: options temperature).
			seed := options seed ifNil: [ 16rFFFFFFFF ].
			lib
				llamaSamplerChainAdd: chain
				sampler: (lib llamaSamplerInitDist: seed) ]
		ifFalse: [
			lib
				llamaSamplerChainAdd: chain
				sampler: lib llamaSamplerInitGreedy ].

	^ chain
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

	| lib buffer batch rc |
	lib := self library.
	buffer := FFIExternalArray
		externalNewType: 'int32'
		size: tokenArray size.
	[
		1 to: tokenArray size do: [ :i |
			buffer at: i put: (tokenArray at: i) ].
		batch := lib
			llamaBatchGetOne: buffer getHandle
			nTokens: tokenArray size.
		rc := lib llamaDecode: ctxPtr batch: batch.
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

	tokens := self
		tokenizeString: text
		vocab: vocabPtr
		addSpecial: true.

	self decodeTokens: tokens context: handle ctxPtr.

	dim := self library llamaModelNEmbd: handle modelPtr.
	embPtr := self library llamaGetEmbeddings: handle ctxPtr.
	embPtr isNull ifTrue: [
		Error signal:
			'llama_get_embeddings returned NULL; create the backend with #embeddings: true to enable pooled embeddings.' ].

	embd := Array new: dim.
	1 to: dim do: [ :i |
		embd at: i put: (embPtr float32AtOffset: (i - 1) * 4) ].
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

	| modelParams ctxParams modelPtr ctxPtr handle path |
	(loadedModels includesKey: aModel name) ifTrue: [ ^ self ].

	aModel path ifNil: [ Error signal: 'Model has no path: ', aModel name ].
	aModel path exists ifFalse: [
		Error signal: 'Model file not found: ', aModel path fullName ].

	self ensureBackendInitialized.

	path := aModel path fullName.

	modelParams := self library llamaModelDefaultParams.
	modelParams nGpuLayers: self nGpuLayers.

	modelPtr := self library
		llamaModelLoadFromFile: path
		params: modelParams.
	modelPtr isNull ifTrue: [
		Error signal: 'llama_model_load_from_file failed for: ', path ].

	ctxParams := self library llamaContextDefaultParams.
	ctxParams
		nCtx: self contextSize;
		nBatch: 512;
		nThreads: self nThreads;
		nThreadsBatch: self nThreads;
		embeddings: self embeddings.

	ctxPtr := self library
		llamaInitFromModel: modelPtr
		params: ctxParams.
	ctxPtr isNull ifTrue: [
		self library llamaModelFree: modelPtr.
		Error signal: 'llama_init_from_model failed for: ', path ].

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

	^ nThreads ifNil: [ Smalltalk vm parameterAt: 58 ifAbsent: [ 4 ] ]
]

{ #category : 'accessing' }
AILocalBackend >> nThreads: anInteger [

	nThreads := anInteger
]

{ #category : 'operations' }
AILocalBackend >> streamText: prompt model: aModel options: options onToken: aBlock [

	| handle vocabPtr lib tokens sampler generated maxTokens nextToken piece |
	self ensureModelLoaded: aModel.
	handle := loadedModels at: aModel name.
	lib := self library.

	vocabPtr := lib llamaModelGetVocab: handle modelPtr.
	tokens := self tokenizeString: prompt vocab: vocabPtr addSpecial: true.
	self decodeTokens: tokens context: handle ctxPtr.

	sampler := self buildSampler: options.
	maxTokens := options maxTokens.
	generated := 0.

	[
		[ generated < maxTokens ] whileTrue: [
			nextToken := lib
				llamaSamplerSample: sampler
				context: handle ctxPtr
				idx: -1.
			(lib llamaVocabIsEog: vocabPtr token: nextToken) ifTrue: [ ^ self ].

			lib llamaSamplerAccept: sampler token: nextToken.
			piece := self detokenize: nextToken vocab: vocabPtr.
			piece ifNotEmpty: [ aBlock value: piece ].

			self
				decodeTokens: (Array with: nextToken)
				context: handle ctxPtr.
			generated := generated + 1 ] ]
		ensure: [ lib llamaSamplerFree: sampler ]
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
			text: aString
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
				text: aString
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
