"
""I represent a chat completion request.
I store the target model, the message history, generation options, and whether the response should be streamed.""
"
Class {
	#name : 'AIChatCompletionRequest',
	#superclass : 'Object',
	#instVars : [
		'model',
		'messages',
		'options',
		'stream'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'as yet unclassified' }
AIChatCompletionRequest class >> model: modelName messages: messagesArray [
	"Create a chat completion request"
	^ self new
		model: modelName;
		messages: messagesArray;
		options: AIGenerationOptions default;
		yourself
]

{ #category : 'accessing' }
AIChatCompletionRequest >> messages [

	^ messages
]

{ #category : 'accessing' }
AIChatCompletionRequest >> messages: anObject [

	messages := anObject
]

{ #category : 'accessing' }
AIChatCompletionRequest >> model [

	^ model
]

{ #category : 'accessing' }
AIChatCompletionRequest >> model: anObject [

	model := anObject
]

{ #category : 'accessing' }
AIChatCompletionRequest >> options [

	^ options ifNil: [ options := AIGenerationOptions default ]
]

{ #category : 'accessing' }
AIChatCompletionRequest >> options: anObject [
	options := anObject
]

{ #category : 'accessing' }
AIChatCompletionRequest >> stream [

	^ stream ifNil: [ false ]
]

{ #category : 'accessing' }
AIChatCompletionRequest >> stream: anObject [

	stream := anObject
]
