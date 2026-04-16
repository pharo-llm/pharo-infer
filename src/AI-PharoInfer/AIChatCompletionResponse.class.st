"
""I represent the result of a chat completion request.
I store the generated assistant message together with metadata such as id, model, finish reason, usage, and creation time.""
"
Class {
	#name : 'AIChatCompletionResponse',
	#superclass : 'Object',
	#instVars : [
		'id',
		'model',
		'message',
		'finishReason',
		'usage',
		'created'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'instance creation' }
AIChatCompletionResponse class >> fromMessage: aMessage model: modelName [

	"Create a response from a message"
	^ self new
		id: UUID new asString;
		model: modelName;
		message: aMessage;
		finishReason: #stop;
		created: DateAndTime now;
		yourself
]

{ #category : 'accessing' }
AIChatCompletionResponse >> created [
	^ created
]

{ #category : 'accessing' }
AIChatCompletionResponse >> created: anObject [

	created := anObject
]

{ #category : 'accessing' }
AIChatCompletionResponse >> finishReason [ 

	^ finishReason

]

{ #category : 'accessing' }
AIChatCompletionResponse >> finishReason: anObject [

	finishReason := anObject
]

{ #category : 'accessing' }
AIChatCompletionResponse >> id [

	^ id
]

{ #category : 'accessing' }
AIChatCompletionResponse >> id: anObject [

	id := anObject
]

{ #category : 'accessing' }
AIChatCompletionResponse >> message [ 

	^ message
]

{ #category : 'accessing' }
AIChatCompletionResponse >> message: anObject [

	message := anObject
]

{ #category : 'accessing' }
AIChatCompletionResponse >> model [

	^ model
]

{ #category : 'accessing' }
AIChatCompletionResponse >> model: anObject [

	model := anObject
]

{ #category : 'accessing' }
AIChatCompletionResponse >> usage [

	^ usage
]

{ #category : 'accessing' }
AIChatCompletionResponse >> usage: anObject [

	usage := anObject
]
