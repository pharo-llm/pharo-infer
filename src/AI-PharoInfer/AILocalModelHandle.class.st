"
I am a lightweight container for the two opaque pointers llama.cpp
hands us when a model is loaded: the model itself and an inference
context. I also remember the source file path so callers can inspect
what was loaded.

I know how to release the underlying C resources via #freeWith:.
"
Class {
	#name : 'AILocalModelHandle',
	#superclass : 'Object',
	#instVars : [
		'modelPtr',
		'ctxPtr',
		'path'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AILocalModelHandle >> ctxPtr [
	^ ctxPtr
]

{ #category : 'accessing' }
AILocalModelHandle >> ctxPtr: anObject [
	ctxPtr := anObject
]

{ #category : 'lifecycle' }
AILocalModelHandle >> freeWith: aLibrary [

	ctxPtr ifNotNil: [
		ctxPtr isNull ifFalse: [ aLibrary llamaFree: ctxPtr ].
		ctxPtr := nil ].
	modelPtr ifNotNil: [
		modelPtr isNull ifFalse: [ aLibrary llamaModelFree: modelPtr ].
		modelPtr := nil ]
]

{ #category : 'accessing' }
AILocalModelHandle >> modelPtr [
	^ modelPtr
]

{ #category : 'accessing' }
AILocalModelHandle >> modelPtr: anObject [
	modelPtr := anObject
]

{ #category : 'accessing' }
AILocalModelHandle >> path [
	^ path
]

{ #category : 'accessing' }
AILocalModelHandle >> path: anObject [
	path := anObject
]
