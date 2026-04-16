"
""I represent an AI model loaded from a file or registered by name.
I store metadata such as path, format, size, backend, and loaded state, and I delegate load/unload behavior to the backend.""
"
Class {
	#name : 'AIModel',
	#superclass : 'Object',
	#instVars : [
		'name',
		'path',
		'format',
		'size',
		'parameters',
		'quantization',
		'contextLength',
		'metadata',
		'loaded',
		'backend'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'instance creation' }
AIModel class >> fromFile: aFileReference [

	| fileReference |
	fileReference := aFileReference asFileReference.
	
	"Create a model from a file"
	^ self new
		path: fileReference;
		name: fileReference basenameWithoutExtension;
		format: (AIModelFormat detectFromFile: fileReference);
		size: fileReference size;
		yourself
]

{ #category : 'instance creation' }
AIModel class >> named: aString [

	"Create a model with a given name"
	^ self new
		name: aString;
		yourself
]

{ #category : 'accessing' }
AIModel >> backend [

	^ backend
]

{ #category : 'accessing' }
AIModel >> backend: anObject [

	backend := anObject
]

{ #category : 'accessing' }
AIModel >> contextLength [

	^ contextLength ifNil: [ 2048 ]
]

{ #category : 'accessing' }
AIModel >> contextLength: anObject [

	contextLength := anObject
]

{ #category : 'accessing' }
AIModel >> format [

	^ format
]

{ #category : 'accessing' }
AIModel >> format: anObject [

	format := anObject
]

{ #category : 'accessing' }
AIModel >> initialize [

	super initialize.
	loaded := false.
	parameters := Dictionary new.
	metadata := Dictionary new
]

{ #category : 'accessing' }
AIModel >> isLoaded [

	^ loaded
]

{ #category : 'accessing' }
AIModel >> load [

	"Load the model into memory"
	self isLoaded ifTrue: [ ^ self ].

	self path ifNil: [
		Error signal: 'Cannot load model: path is not set' ].

	self backend ifNil: [
		Error signal: 'Cannot load model: backend is not set' ].

	self backend loadModel: self.
	loaded := true
]

{ #category : 'accessing' }
AIModel >> metadata [

	^ metadata
]

{ #category : 'accessing' }
AIModel >> metadata: anObject [

	metadata := anObject
]

{ #category : 'accessing' }
AIModel >> name [ 

	^ name
]

{ #category : 'accessing' }
AIModel >> name: anObject [

	name := anObject
]

{ #category : 'accessing' }
AIModel >> parameters [

	^ parameters
]

{ #category : 'accessing' }
AIModel >> parameters: anObject [ 

	parameters := anObject
]

{ #category : 'accessing' }
AIModel >> path [

	^ path
]

{ #category : 'accessing' }
AIModel >> path: anObject [

	path := anObject
]

{ #category : 'accessing' }
AIModel >> printOn: aStream [

	super printOn: aStream.
	aStream
		nextPutAll: '(';
		nextPutAll: (name ifNil: [ 'unnamed' ]);
		nextPutAll: ', ';
		nextPutAll: (self isLoaded ifTrue: [ 'loaded' ] ifFalse: [ 'unloaded' ]);
		nextPutAll: ')'
]

{ #category : 'accessing' }
AIModel >> quantization [

	^ quantization
]

{ #category : 'accessing' }
AIModel >> quantization: anObject [

	quantization := anObject
]

{ #category : 'accessing' }
AIModel >> size [ 

	^ size
]

{ #category : 'accessing' }
AIModel >> size: anObject [

	size := anObject
]

{ #category : 'accessing' }
AIModel >> unload [

	"Unload the model from memory"
	self isLoaded ifFalse: [ ^ self ].

	self backend ifNotNil: [
		self backend unloadModel: self ].

	loaded := false
]
