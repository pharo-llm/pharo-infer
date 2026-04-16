"
""I provide a chat-oriented API on top of the inference engine.
I transform structured chat messages into a prompt, invoke completion or streaming, and wrap results into chat response objects.""
"
Class {
	#name : 'AIChatAPI',
	#superclass : 'Object',
	#instVars : [
		'inferenceEngine'
	],
	#classInstVars : [
		'default'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AIChatAPI class >> default [

	^ default ifNil: [ default := self new ]
]

{ #category : 'accessing' }
AIChatAPI class >> reset [
	<script>
	default := nil
]

{ #category : 'as yet unclassified' }
AIChatAPI >> complete: request [
	"Complete a chat request and return a response"
	| prompt response responseMessage |

	"Format messages into a prompt"
	prompt := self formatMessages: request messages.

	"Generate completion"
	request stream ifTrue: [
		Error signal: 'Streaming is not supported in this method. Use streamComplete:onToken:' ].

	response := self inferenceEngine
		complete: prompt
		model: request model
		options: request options.

	"Create response message"
	responseMessage := AIChatMessage assistant: response.

	^ AIChatCompletionResponse
		fromMessage: responseMessage
		model: request model
]

{ #category : 'private' }
AIChatAPI >> formatMessages: messages [
	"Format chat messages into a single prompt string"
	| stream |

	stream := WriteStream on: String new.

	messages do: [ :msg |
		msg role = #system ifTrue: [
			stream
				nextPutAll: 'System: ';
				nextPutAll: msg content;
				cr ].

		msg role = #user ifTrue: [
			stream
				nextPutAll: 'User: ';
				nextPutAll: msg content;
				cr ].

		msg role = #assistant ifTrue: [
			stream
				nextPutAll: 'Assistant: ';
				nextPutAll: msg content;
				cr ] ].

	stream nextPutAll: 'Assistant: '.

	^ stream contents
]

{ #category : 'private' }
AIChatAPI >> inferenceEngine [

	^ inferenceEngine ifNil: [ inferenceEngine := AIInferenceEngine default ]
]

{ #category : 'private' }
AIChatAPI >> inferenceEngine: anObject [

	inferenceEngine := anObject
]

{ #category : 'private' }
AIChatAPI >> streamComplete: request onToken: aBlock [

	"Stream a chat completion, calling aBlock for each token"
	| prompt |

	"Format messages into a prompt"
	prompt := self formatMessages: request messages.

	"Stream generation"
	self inferenceEngine
		stream: prompt
		model: request model
		options: request options
		onToken: aBlock
]
