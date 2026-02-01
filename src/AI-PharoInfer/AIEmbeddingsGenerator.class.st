Class {
	#name : 'AIEmbeddingsGenerator',
	#superclass : 'Object',
	#instVars : [
		'model',
		'backend',
		'dimensions'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'as yet unclassified' }
AIEmbeddingsGenerator class >> forModel: aModel [

	"Create an embeddings generator for a specific model"
	^ self new
		model: aModel;
		yourself
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> backend [

	^ backend ifNil: [ backend := AILocalBackend new ]
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> backend: anObject [

	backend := anObject
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> cosineSimilarity: embedding1 with: embedding2 [ 

	"Compute cosine similarity between two embeddings"
	| dotProduct norm1 norm2 |

	dotProduct := 0.0.
	norm1 := 0.0.
	norm2 := 0.0.

	1 to: embedding1 size do: [ :i |
		| v1 v2 |
		v1 := embedding1 at: i.
		v2 := embedding2 at: i.
		dotProduct := dotProduct + (v1 * v2).
		norm1 := norm1 + (v1 * v1).
		norm2 := norm2 + (v2 * v2) ].

	norm1 := norm1 sqrt.
	norm2 := norm2 sqrt.

	(norm1 = 0 or: [ norm2 = 0 ]) ifTrue: [ ^ 0.0 ].

	^ dotProduct / (norm1 * norm2)
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> dimensions [

	^ dimensions ifNil: [ 768 ]
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> dimensions: anObject [

	dimensions := anObject
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> embed: text [

	"Generate embeddings for a single text"
	^ self backend generateEmbeddings: text model: self model
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> embedBatch: texts [

	"Generate embeddings for multiple texts"
	^ texts collect: [ :text |
		self embed: text ]
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> model [

	^ model
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> model: anObject [

	model := anObject
]

{ #category : 'accessing' }
AIEmbeddingsGenerator >> normalize: embedding [

	"Normalize an embedding to unit length"
	| norm normalized |

	norm := 0.0.
	embedding do: [ :value |
		norm := norm + (value * value) ].
	norm := norm sqrt.

	norm = 0 ifTrue: [ ^ embedding ].

	normalized := embedding collect: [ :value |
		value / norm ].

	^ normalized
]
