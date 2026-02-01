Class {
	#name : 'AIInferenceEngineTest',
	#superclass : 'TestCase',
	#instVars : [
		'engine',
		'modelManager',
		'testModel'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIInferenceEngineTest >> setUp [
	super setUp.

	"Create a test model"
	testModel := AIModel named: 'test-model'.
	testModel backend: AILocalBackend new.
	testModel path: (FileLocator temp / 'test-model.gguf').
	testModel path ensureCreateFile.

	"Setup model manager"
	modelManager := AIModelManager new.
	modelManager registerModel: testModel.

	"Setup engine"
	engine := AIInferenceEngine new.
	engine modelManager: modelManager
]

{ #category : 'running' }
AIInferenceEngineTest >> tearDown [

	testModel path ensureDelete.
	super tearDown
]

{ #category : 'running' }
AIInferenceEngineTest >> testComplete [

	| result |

	result := engine complete: 'Hello' model: 'test-model'.

	self assert: result isString.
	self assert: result notEmpty
]

{ #category : 'running' }
AIInferenceEngineTest >> testCompleteWithOptions [

	| result options |

	options := AIGenerationOptions new
		temperature: 0.5;
		maxTokens: 100;
		yourself.

	result := engine complete: 'Hello' model: 'test-model' options: options.

	self assert: result isString.
	self assert: result notEmpty
]

{ #category : 'running' }
AIInferenceEngineTest >> testEmbeddings [

	| result |

	result := engine embeddings: 'test text' model: 'test-model'.

	self assert: result notNil
]

{ #category : 'running' }
AIInferenceEngineTest >> testStream [

	| tokens |

	tokens := OrderedCollection new.

	engine
		stream: 'Hello'
		model: 'test-model'
		onToken: [ :token | tokens add: token ].

	"Should have generated some tokens"
	self assert: tokens notEmpty
]
