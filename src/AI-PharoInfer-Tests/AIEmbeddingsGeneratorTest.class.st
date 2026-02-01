Class {
	#name : 'AIEmbeddingsGeneratorTest',
	#superclass : 'TestCase',
	#instVars : [
		'generator',
		'testModel'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> setUp [
	super setUp.

	testModel := AIModel named: 'test-embeddings-model'.
	testModel backend: AILocalBackend new.
	testModel path: (FileLocator temp / 'test-embeddings.gguf').
	testModel path ensureCreateFile.

	generator := AIEmbeddingsGenerator forModel: testModel
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> tearDown [
	testModel path ensureDelete.
	super tearDown
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> testCosineSimilarity [
	| emb1 emb2 similarity |

	emb1 := #(1.0 0.0 0.0).
	emb2 := #(1.0 0.0 0.0).

	similarity := generator cosineSimilarity: emb1 with: emb2.

	self assert: (similarity closeTo: 1.0 precision: 0.01)
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> testCosineSimilarityOrthogonal [

	| emb1 emb2 similarity |

	emb1 := #(1.0 0.0 0.0).
	emb2 := #(0.0 1.0 0.0).

	similarity := generator cosineSimilarity: emb1 with: emb2.

	self assert: (similarity closeTo: 0.0 precision: 0.01)
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> testEmbed [
	| embedding |

	testModel load.
	embedding := generator embed: 'test text'.

	self assert: embedding notNil.
	self assert: embedding notEmpty
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> testEmbedBatch [

	| embeddings texts |

	testModel load.
	texts := #('hello' 'world' 'test').

	embeddings := generator embedBatch: texts.

	self assert: embeddings size equals: 3.
	embeddings do: [ :emb |
		self assert: emb notNil ]
]

{ #category : 'running' }
AIEmbeddingsGeneratorTest >> testNormalize [

	| embedding normalized magnitude |

	embedding := #(3.0 4.0).
	normalized := generator normalize: embedding.

	"Calculate magnitude of normalized vector"
	magnitude := ((normalized first squared) + (normalized second squared)) sqrt.

	self assert: (magnitude closeTo: 1.0 precision: 0.001)
]
