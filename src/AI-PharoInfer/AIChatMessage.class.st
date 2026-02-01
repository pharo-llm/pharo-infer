Class {
	#name : 'AIChatMessage',
	#superclass : 'Object',
	#instVars : [
		'role',
		'content',
		'name'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'as yet unclassified' }
AIChatMessage class >> assistant: content [

	"Create an assistant message"
	^ self new
		role: #assistant;
		content: content;
		yourself
]

{ #category : 'as yet unclassified' }
AIChatMessage class >> system: content [
	"Create a system message"
	^ self new
		role: #system;
		content: content;
		yourself
]

{ #category : 'as yet unclassified' }
AIChatMessage class >> user: content [

	"Create a user message"
	^ self new
		role: #user;
		content: content;
		yourself
]

{ #category : 'converting' }
AIChatMessage >> asString [

	^ String streamContents: [ :stream |
		stream
			nextPutAll: '<';
			nextPutAll: role asString;
			nextPutAll: '>: ';
			nextPutAll: content ]
]

{ #category : 'converting' }
AIChatMessage >> content [

	^ content
]

{ #category : 'converting' }
AIChatMessage >> content: anObject [

	content := anObject
]

{ #category : 'converting' }
AIChatMessage >> name [

	^ name
]

{ #category : 'converting' }
AIChatMessage >> name: anObject [

	name := anObject
]

{ #category : 'printing' }
AIChatMessage >> printOn: aStream [

	super printOn: aStream.
	aStream
		nextPutAll: '(';
		nextPutAll: role asString;
		nextPutAll: ': ';
		nextPutAll: (content copyFrom: 1 to: (content size min: 50));
		nextPutAll: (content size > 50 ifTrue: [ '...' ] ifFalse: [ '' ]);
		nextPutAll: ')'
]

{ #category : 'printing' }
AIChatMessage >> role [

	^ role
]

{ #category : 'printing' }
AIChatMessage >> role: anObject [

	role := anObject
]
