"
I describe a supported model file format. The local llama.cpp backend
only accepts GGUF, so that is the only concrete format exposed. Unknown
files are still representable so callers can report a clean error.
"
Class {
	#name : 'AIModelFormat',
	#superclass : 'Object',
	#instVars : [
		'name',
		'extension',
		'description'
	],
	#classInstVars : [
		'formats'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'instance creation' }
AIModelFormat class >> detectFromFile: aFileReference [

	| ext |
	ext := aFileReference extension.
	^ self formats
		detect: [ :format | format extension = ext ]
		ifNone: [ self unknown ]
]

{ #category : 'instance creation' }
AIModelFormat class >> formats [

	^ formats ifNil: [ formats := self initializeFormats ]
]

{ #category : 'instance creation' }
AIModelFormat class >> gguf [

	^ self named: 'GGUF' extension: 'gguf' description: 'GPT-Generated Unified Format for llama.cpp'
]

{ #category : 'instance creation' }
AIModelFormat class >> initializeFormats [

	^ OrderedCollection with: self gguf
]

{ #category : 'instance creation' }
AIModelFormat class >> named: aName extension: anExtension description: aDescription [

	^ self new
		name: aName;
		extension: anExtension;
		description: aDescription;
		yourself
]

{ #category : 'instance creation' }
AIModelFormat class >> unknown [

	^ self named: 'Unknown' extension: '' description: 'Unknown model format'
]

{ #category : 'accessing' }
AIModelFormat >> description [

	^ description
]

{ #category : 'accessing' }
AIModelFormat >> description: anObject [

	description := anObject
]

{ #category : 'accessing' }
AIModelFormat >> extension [

	^ extension
]

{ #category : 'accessing' }
AIModelFormat >> extension: anObject [

	extension := anObject
]

{ #category : 'accessing' }
AIModelFormat >> name [

	^ name
]

{ #category : 'accessing' }
AIModelFormat >> name: anObject [

	name := anObject
]

{ #category : 'printing' }
AIModelFormat >> printOn: aStream [

	super printOn: aStream.
	aStream
		nextPutAll: '(';
		nextPutAll: name;
		nextPutAll: ')'
]
