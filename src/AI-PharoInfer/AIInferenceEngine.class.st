Class {
	#name : 'AIInferenceEngine',
	#superclass : 'Object',
	#instVars : [
		'modelManager',
		'backend'
	],
	#classInstVars : [
		'default'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AIInferenceEngine class >> default [

	^ default ifNil: [ default := self new ]
]

{ #category : 'accessing' }
AIInferenceEngine class >> reset [

	default := nil
]

{ #category : 'accessing' }
AIInferenceEngine >> backend [

	^ backend ifNil: [ backend := AILocalBackend new ]
]

{ #category : 'accessing' }
AIInferenceEngine >> backend: anObject [

	backend := anObject
]

{ #category : 'accessing' }
AIInferenceEngine >> complete: prompt model: modelName [

	"Generate text completion for the given prompt"
	^ self complete: prompt model: modelName options: AIGenerationOptions default
]

{ #category : 'accessing' }
AIInferenceEngine >> complete: prompt model: modelName options: options [

	"Generate text completion with custom options"
	| model result |

	model := self resolveModelNamed: modelName.
	model ifNil: [
		Error signal: 'Model not found: ', modelName ].

	model isLoaded ifFalse: [
		model load ].

	result := self backend generateText: prompt model: model options: options.

	^ result
]

{ #category : 'accessing' }
AIInferenceEngine >> embeddings: text model: modelName [

	"Generate embeddings for the given text"
	| model result |

	model := self resolveModelNamed: modelName.
	model ifNil: [
		Error signal: 'Model not found: ', modelName ].

	model isLoaded ifFalse: [
		model load ].

	result := self backend generateEmbeddings: text model: model.

	^ result
]

{ #category : 'accessing' }
AIInferenceEngine >> initialize [

	super initialize.
	modelManager := AIModelManager default
]

{ #category : 'accessing' }
AIInferenceEngine >> modelManager [

	^ modelManager
]

{ #category : 'accessing' }
AIInferenceEngine >> modelManager: anObject [

	modelManager := anObject
]

{ #category : 'private' }
AIInferenceEngine >> resolveModelNamed: modelName [

	| model defaultManager |
	model := self modelManager modelNamed: modelName.
	model ifNotNil: [ ^ model ].

	defaultManager := AIModelManager default.
	(defaultManager ~~ self modelManager)
		ifTrue: [
			model := defaultManager modelNamed: modelName.
			model ifNotNil: [
				self modelManager: defaultManager ] ].

	^ model
]

{ #category : 'accessing' }
AIInferenceEngine >> stream: prompt model: modelName onToken: aBlock [ 

	"Stream text generation, calling aBlock for each generated token"
	^ self stream: prompt model: modelName options: AIGenerationOptions default onToken: aBlock
]

{ #category : 'accessing' }
AIInferenceEngine >> stream: prompt model: modelName options: options onToken: aBlock [

	"Stream text generation with custom options"
	| model |

	model := self resolveModelNamed: modelName.
	model ifNil: [
		Error signal: 'Model not found: ', modelName ].

	model isLoaded ifFalse: [
		model load ].

	self backend streamText: prompt model: model options: options onToken: aBlock
]
