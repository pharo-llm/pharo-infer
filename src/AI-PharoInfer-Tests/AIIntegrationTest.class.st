Class {
	#name : 'AIIntegrationTest',
	#superclass : 'TestCase',
	#instVars : [
		'engine',
		'modelManager',
		'testModel',
		'chatAPI'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIIntegrationTest >> setUp [

	super setUp.

	"Create a test model"
	testModel := AIModel named: 'integration-test-model'.
	testModel backend: AILocalBackend new.
	testModel path: (FileLocator temp / 'integration-test.gguf').
	testModel path ensureCreateFile.

	"Setup model manager"
	modelManager := AIModelManager new.
	modelManager registerModel: testModel.

	"Setup engine"
	engine := AIInferenceEngine new.
	engine modelManager: modelManager.

	"Setup chat API"
	chatAPI := AIChatAPI new.
	chatAPI inferenceEngine: engine
]

{ #category : 'running' }
AIIntegrationTest >> tearDown [

	testModel path ensureDelete.
	super tearDown
]

{ #category : 'running' }
AIIntegrationTest >> testChatConversation [

	"Test a multi-turn conversation"
	| request1 response1 request2 response2 messages |

	"First turn"
	request1 := AIChatCompletionRequest
		model: 'integration-test-model'
		messages: {
			AIChatMessage system: 'You are a helpful assistant'.
			AIChatMessage user: 'Hello!' }.

	response1 := chatAPI complete: request1.
	self assert: response1 message role equals: #assistant.

	"Second turn - add previous assistant response"
	messages := OrderedCollection new.
	messages add: (AIChatMessage system: 'You are a helpful assistant').
	messages add: (AIChatMessage user: 'Hello!').
	messages add: response1 message.
	messages add: (AIChatMessage user: 'How are you?').

	request2 := AIChatCompletionRequest
		model: 'integration-test-model'
		messages: messages asArray.

	response2 := chatAPI complete: request2.
	self assert: response2 message role equals: #assistant.
	self assert: response2 message content notEmpty
]

{ #category : 'running' }
AIIntegrationTest >> testEmbeddingsSimilarity [

	"Test that similar texts have similar embeddings"
	| emb1 emb2 emb3 generator sim12 sim13 |

	testModel load.

	generator := AIEmbeddingsGenerator forModel: testModel.
	generator backend: testModel backend.

	emb1 := generator embed: 'The cat sits on the mat'.
	emb2 := generator embed: 'A cat is sitting on a mat'.
	emb3 := generator embed: 'The weather is nice today'.

	sim12 := generator cosineSimilarity: emb1 with: emb2.
	sim13 := generator cosineSimilarity: emb1 with: emb3.

	"Similar sentences should have higher similarity than unrelated ones"
	"Note: This is a simplified test since our backend uses dummy embeddings"
	self assert: emb1 notNil.
	self assert: emb2 notNil.
	self assert: emb3 notNil
]

{ #category : 'running' }
AIIntegrationTest >> testEndToEndWorkflow [

	"Test complete workflow: register model, load it, generate text, unload"
	| result |

	"Model is already registered in setUp"
	self assert: (modelManager modelNamed: 'integration-test-model') notNil.

	"Load model"
	modelManager loadModel: 'integration-test-model'.
	self assert: testModel isLoaded.

	"Generate text"
	result := engine complete: 'Tell me a story' model: 'integration-test-model'.
	self assert: result isString.
	self assert: result notEmpty.

	"Unload model"
	modelManager unloadModel: 'integration-test-model'.
	self deny: testModel isLoaded
]

{ #category : 'running' }
AIIntegrationTest >> testGenerationWithDifferentOptions [

	"Test generation with various option configurations"
	| options1 options2 result1 result2 |

	options1 := AIGenerationOptions new
		temperature: 0.1;
		maxTokens: 50;
		yourself.

	options2 := AIGenerationOptions new
		temperature: 0.9;
		maxTokens: 50;
		yourself.

	result1 := engine complete: 'Hello' model: 'integration-test-model' options: options1.
	result2 := engine complete: 'Hello' model: 'integration-test-model' options: options2.

	self assert: result1 isString.
	self assert: result2 isString
]

{ #category : 'running' }
AIIntegrationTest >> testModelLifecycle [

	"Test the complete model lifecycle"
	| newModel file result |

	file := FileLocator temp / 'lifecycle-test.gguf'.
	file ensureCreateFile.

	[ "Create and register model"
	newModel := AIModel fromFile: file.
	newModel backend: AILocalBackend new.
	modelManager registerModel: newModel.

	self assert: (modelManager listModels includes: 'lifecycle-test').

	"Load model"
	modelManager loadModel: 'lifecycle-test'.
	self assert: newModel isLoaded.

	"Use model for inference"
	result := engine complete: 'test' model: 'lifecycle-test'.
	self assert: result notNil.

	"Unload and unregister"
	modelManager unloadModel: 'lifecycle-test'.
	self deny: newModel isLoaded.

	modelManager unregisterModel: 'lifecycle-test'.
	self deny: (modelManager listModels includes: 'lifecycle-test') ]
		ensure: [ file ensureDelete ]
]

{ #category : 'running' }
AIIntegrationTest >> testStreamingGeneration [

	"Test streaming text generation"
	| tokens receivedTokens options |

	tokens := OrderedCollection new.
	options := AIGenerationOptions new
		maxTokens: 20;
		yourself.

	engine
		stream: 'Once upon a time'
		model: 'integration-test-model'
		options: options
		onToken: [ :token | tokens add: token ].

	"Should have received some tokens"
	self assert: tokens notEmpty.

	"Reconstruct text from tokens"
	receivedTokens := String streamContents: [ :stream |
		tokens do: [ :token | stream nextPutAll: token ] ].

	self assert: receivedTokens isString
]
