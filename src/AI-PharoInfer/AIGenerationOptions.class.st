Class {
	#name : 'AIGenerationOptions',
	#superclass : 'Object',
	#instVars : [
		'temperature',
		'topP',
		'topK',
		'maxTokens',
		'stopSequences',
		'repeatPenalty',
		'frequencyPenalty',
		'presencePenalty',
		'seed',
		'stream'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AIGenerationOptions class >> default [

	"Return default generation options"
	^ self new
		temperature: 0.7;
		topP: 0.9;
		topK: 40;
		maxTokens: 512;
		repeatPenalty: 1.1;
		stream: false;
		yourself
]

{ #category : 'accessing' }
AIGenerationOptions >> frequencyPenalty [

	^ frequencyPenalty ifNil: [ 0.0 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> frequencyPenalty: anObject [

	frequencyPenalty := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> initialize [

	super initialize.
	stopSequences := OrderedCollection new.
	stream := false
]

{ #category : 'accessing' }
AIGenerationOptions >> maxTokens [

	^ maxTokens ifNil: [ 512 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> maxTokens: anObject [

	maxTokens := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> presencePenalty [

	^ presencePenalty ifNil: [ 0.0 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> presencePenalty: anObject [

	presencePenalty := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> repeatPenalty [

	^ repeatPenalty ifNil: [ 1.1 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> repeatPenalty: anObject [

	repeatPenalty := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> seed [

	^ seed
]

{ #category : 'accessing' }
AIGenerationOptions >> seed: anObject [

	seed := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> stopSequences [

	^ stopSequences
]

{ #category : 'accessing' }
AIGenerationOptions >> stopSequences: anObject [

	stopSequences := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> stream [

	^ stream
]

{ #category : 'accessing' }
AIGenerationOptions >> stream: anObject [

	stream := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> temperature [

	^ temperature ifNil: [ 0.7 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> temperature: anObject [

	temperature := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> topK [

	^ topK ifNil: [ 40 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> topK: anObject [

	topK := anObject
]

{ #category : 'accessing' }
AIGenerationOptions >> topP [

	^ topP ifNil: [ 0.9 ]
]

{ #category : 'accessing' }
AIGenerationOptions >> topP: anObject [

	topP := anObject
]
