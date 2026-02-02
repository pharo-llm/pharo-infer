Class {
	#name : 'AILocalBackend',
	#superclass : 'AIBackend',
	#instVars : [
		'loadedModels',
		'tokenizer'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'private' }
AILocalBackend >> computeEmbeddings: tokens for: aModel [


	"Compute embeddings for tokens (simplified implementation)"
	| modelData embedding dimensions |

	modelData := loadedModels at: aModel name.
	dimensions := 768.

	"Generate a single embedding vector by averaging token embeddings"
	"In a real implementation, this would use the model's embedding layer and pooling"
	embedding := Array new: dimensions.
	1 to: dimensions do: [ :i |
		| sum |
		sum := tokens inject: 0 into: [ :acc :token |
			acc + (token hash + i) ].
		embedding at: i put: (sum / (tokens size * 1000.0)) ].

	^ embedding
]

{ #category : 'private' }
AILocalBackend >> computeLogits: tokens model: aModel [

	"Compute logits for next token prediction (simplified)"
	| vocabSize logits |

	vocabSize := self tokenizer vocabularySize.

	"Generate random logits (in reality, this would be model output)"
	logits := Array new: vocabSize.
	1 to: vocabSize do: [ :i |
		logits at: i put: (Random new next * 10.0) ].

	^ logits
]

{ #category : 'private' }
AILocalBackend >> ensureModelLoaded: aModel [

	"Ensure the model is loaded"
	(loadedModels includesKey: aModel name) ifFalse: [
		Error signal: 'Model not loaded: ', aModel name ]
]

{ #category : 'private' }
AILocalBackend >> generateEmbeddings: text model: aModel [
	"Generate embeddings for the given text using local inference"
	| tokens embedding |

	self ensureModelLoaded: aModel.

	"Tokenize the text"
	tokens := self tokenizer tokenize: text.

	"Generate embeddings (simplified implementation)"
	embedding := self computeEmbeddings: tokens for: aModel.

	^ embedding
]

{ #category : 'private' }
AILocalBackend >> generateText: prompt model: aModel options: options [
	"Generate text from a prompt using local inference"
	| tokens result |

	self ensureModelLoaded: aModel.

	"Tokenize the prompt"
	tokens := self tokenizer tokenize: prompt.

	"Generate tokens"
	result := self performInference: tokens model: aModel options: options.

	^ result
]

{ #category : 'private' }
AILocalBackend >> initialize [

	super initialize.
	loadedModels := Dictionary new.
	tokenizer := AITokenizer new
]

{ #category : 'private' }
AILocalBackend >> isEndToken: token [

	"Check if token is an end-of-sequence token"
	^ token = self tokenizer eosToken
]

{ #category : 'private' }
AILocalBackend >> loadModel: aModel [

	"Load a model into memory"
	| modelData |

	(loadedModels includesKey: aModel name) ifTrue: [ ^ self ].

	"Load model data from file"
	modelData := self loadModelData: aModel.

	loadedModels at: aModel name put: modelData
]

{ #category : 'private' }
AILocalBackend >> loadModelData: aModel [

	"Load model data from file (simplified implementation)"
	| data parser tokenizerData vocab |

	aModel path exists ifFalse: [
		Error signal: 'Model file not found: ', aModel path fullName ].
	
	
	"Parse GGUF file if format is GGUF"
(aModel format notNil and: [ aModel format extension asLowercase = 'gguf' ]) ifTrue: [
		parser := AIGGUFParser parseFile: aModel path.
		tokenizerData := parser tokenizerData.

		"Build vocabulary from tokens array"
		vocab := Dictionary new.
		(tokenizerData at: 'tokens' ifAbsent: [ #() ]) withIndexDo: [ :token :index |
			"GGUF uses 0-based indexing"
			vocab at: token put: (index - 1) ].

		"Create new tokenizer with loaded vocabulary"
		tokenizer := AITokenizer new.
		tokenizer vocabulary: vocab.

		"Set special tokens if available"
		(tokenizerData at: 'bos_token_id' ifAbsent: [ nil ]) ifNotNil: [ :id |
			tokenizer bosToken: id ].
		(tokenizerData at: 'eos_token_id' ifAbsent: [ nil ]) ifNotNil: [ :id |
			tokenizer eosToken: id ].
		(tokenizerData at: 'pad_token_id' ifAbsent: [ nil ]) ifNotNil: [ :id |
			tokenizer padToken: id ].
		(tokenizerData at: 'unk_token_id' ifAbsent: [ nil ]) ifNotNil: [ :id |
			tokenizer unkToken: id ].

		"Store model data"
		data := Dictionary new.
		data at: 'path' put: aModel path.
		data at: 'format' put: aModel format.
		data at: 'loaded_at' put: DateAndTime now.
		data at: 'vocabulary' put: vocab.
		data at: 'metadata' put: parser metadata.

		^ data ].
	
	

	"In a real implementation, this would parse GGUF/safetensors format"
	data := Dictionary new.
	data at: 'path' put: aModel path.
	data at: 'format' put: aModel format.
	data at: 'loaded_at' put: DateAndTime now.

	^ data
]

{ #category : 'private' }
AILocalBackend >> performInference: tokens model: aModel options: options [

	"Perform inference to generate text (simplified implementation)"
	| modelData generated result tokens |

	modelData := loadedModels at: aModel name.

	"Generate tokens (simplified - in reality this would use the model's forward pass)"
	generated := OrderedCollection new.

	options maxTokens timesRepeat: [
		| nextToken |
		nextToken := self predictNextToken: tokens model: aModel options: options.

		(self isEndToken: nextToken) ifTrue: [ ^ self tokenizer decode: generated ].

		generated add: nextToken.
		tokens := tokens copyWith: nextToken ].

	result := self tokenizer decode: generated.
	^ result
]

{ #category : 'private' }
AILocalBackend >> predictNextToken: tokens model: aModel options: options [

	"Predict next token (simplified implementation using sampling)"
	| logits sampledToken |

	"In a real implementation, this would:
	1. Run the model's forward pass
	2. Apply temperature scaling
	3. Apply top-p/top-k sampling
	4. Return the sampled token"

	"For now, return a dummy token based on context"
	logits := self computeLogits: tokens model: aModel.

	sampledToken := self sampleToken: logits options: options.

	^ sampledToken
]

{ #category : 'private' }
AILocalBackend >> sampleFromDistribution: probs [

	"Sample an index from a probability distribution"
	| rand cumulative |

	rand := Random new next.
	cumulative := 0.0.

	probs withIndexDo: [ :prob :idx |
		cumulative := cumulative + prob.
		rand <= cumulative ifTrue: [ ^ idx ] ].

	^ probs size
]

{ #category : 'private' }
AILocalBackend >> sampleToken: logits options: options [

	"Sample a token from logits using temperature and top-p"
	| temperature probs sampledIdx sum |

	temperature := options temperature.

	"Apply temperature scaling"
	probs := logits collect: [ :logit | (logit / temperature) exp ].

	"Normalize probabilities"
	sum := probs inject: 0 into: [ :acc :p | acc + p ].
	probs := probs collect: [ :p | p / sum ].

	"Sample from distribution (simplified - should implement top-p/top-k)"
	sampledIdx := self sampleFromDistribution: probs.

	"Convert 1-based array index to 0-based token ID (GGUF convention)"
	^ sampledIdx - 1
]

{ #category : 'private' }
AILocalBackend >> streamText: prompt model: aModel options: options onToken: aBlock [

	"Stream text generation, calling aBlock for each token"
	| tokens generatedTokens |

	self ensureModelLoaded: aModel.

	"Tokenize the prompt"
	tokens := self tokenizer tokenize: prompt.

	"Generate tokens one by one"
	generatedTokens := 0.
	[ generatedTokens < options maxTokens ] whileTrue: [
		| nextToken |
		nextToken := self predictNextToken: tokens model: aModel options: options.

		"Check for end of sequence"
		(self isEndToken: nextToken) ifTrue: [ ^ self ].

		"Add token to context"
		tokens := tokens copyWith: nextToken.

		"Decode and send to block"
		aBlock value: (self tokenizer decode: nextToken).

		generatedTokens := generatedTokens + 1 ]
]

{ #category : 'private' }
AILocalBackend >> tokenizer [

	^ tokenizer
]

{ #category : 'private' }
AILocalBackend >> tokenizer: anObject [

	tokenizer := anObject
]

{ #category : 'private' }
AILocalBackend >> unloadModel: aModel [

	"Unload a model from memory"
	loadedModels removeKey: aModel name ifAbsent: [ ]
]
