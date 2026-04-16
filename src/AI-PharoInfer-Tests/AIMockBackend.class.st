"
I am a mock backend used by the unit tests. I do not touch any
external library and always return deterministic canned output.
"
Class {
	#name : 'AIMockBackend',
	#superclass : 'AIBackend',
	#category : 'AI-PharoInfer-Tests-Mock',
	#package : 'AI-PharoInfer-Tests',
	#tag : 'Mock'
}

{ #category : 'operations' }
AIMockBackend >> generateEmbeddings: text model: aModel [

	| hash dimensions embedding |

	dimensions := 768.
	hash := text hash abs.

	embedding := Array new: dimensions.
	1 to: dimensions do: [ :i |
		embedding at: i put: ((hash + i) / 1000.0) ].

	^ embedding
]

{ #category : 'operations' }
AIMockBackend >> generateText: prompt model: aModel options: options [

	^ 'Mock response for: ', prompt
]

{ #category : 'operations' }
AIMockBackend >> loadModel: aModel [

	^ self
]

{ #category : 'operations' }
AIMockBackend >> streamText: prompt model: aModel options: options onToken: aBlock [

	| words |
	words := #('Mock' 'streaming' 'response' 'for' 'testing').
	1 to: (options maxTokens min: words size) do: [ :i |
		aBlock value: (words at: i), ' ' ]
]

{ #category : 'operations' }
AIMockBackend >> unloadModel: aModel [

	^ self
]
