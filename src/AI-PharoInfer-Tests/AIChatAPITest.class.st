Class {
	#name : 'AIChatAPITest',
	#superclass : 'TestCase',
	#instVars : [
		'api',
		'testModel',
		'modelManager'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIChatAPITest >> setUp [
	super setUp.

	"Create a test model"
	testModel := AIModel named: 'test-model'.
	testModel backend: AILocalBackend new.
	testModel path: (FileLocator temp / 'test-chat-model.gguf').
	testModel path ensureCreateFile.

	"Setup model manager"
	modelManager := AIModelManager new.
	modelManager registerModel: testModel.

	"Setup API"
	api := AIChatAPI new.
	api inferenceEngine modelManager: modelManager
]

{ #category : 'running' }
AIChatAPITest >> tearDown [

	testModel path ensureDelete.
	super tearDown
]

{ #category : 'running' }
AIChatAPITest >> testChatCompletion [

	| request response |

	request := AIChatCompletionRequest
		model: 'test-model'
		messages: {
			AIChatMessage system: 'You are a helpful assistant'.
			AIChatMessage user: 'Hello!' }.

	response := api complete: request.

	self assert: response notNil.
	self assert: response message notNil.
	self assert: response message role equals: #assistant.
	self assert: response message content isString
]

{ #category : 'running' }
AIChatAPITest >> testFormatMessages [
	| messages formatted |

	messages := {
		AIChatMessage system: 'You are helpful'.
		AIChatMessage user: 'Hi'.
		AIChatMessage assistant: 'Hello!' }.

	formatted := api formatMessages: messages.

	self assert: (formatted includesSubstring: 'System:').
	self assert: (formatted includesSubstring: 'User:').
	self assert: (formatted includesSubstring: 'Assistant:')
]

{ #category : 'running' }
AIChatAPITest >> testStreamCompletion [ 
	| request tokens |

	request := AIChatCompletionRequest
		model: 'test-model'
		messages: { AIChatMessage user: 'Hello!' }.

	tokens := OrderedCollection new.

	api streamComplete: request onToken: [ :token |
		tokens add: token ].

	self assert: tokens notEmpty
]
