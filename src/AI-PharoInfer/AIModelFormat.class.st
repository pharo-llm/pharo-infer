"
""I describe a supported model file format.
Instances define a format name, file extension, and human-readable description, and the class provides helpers for detecting formats from files.""
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

{ #category : 'as yet unclassified' }
AIModelFormat class >> detectFromFile: aFileReference [ 

	"Detect the model format from a file"
	| ext |
	ext := aFileReference extension.
	^ self formats
		detect: [ :format | format extension = ext ]
		ifNone: [ self unknown ]
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> formats [

	"Return all registered formats"
	^ formats ifNil: [ formats := self initializeFormats ]
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> gguf [

	"GGUF format (GPT-Generated Unified Format)"
	^ self named: 'GGUF' extension: 'gguf' description: 'GPT-Generated Unified Format for llama.cpp'
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> initializeFormats [

	"Initialize the list of supported formats"
	^ OrderedCollection with: self gguf with: self safetensors with: self pytorch
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> named: aName extension: anExtension description: aDescription [

	^ self new
		name: aName;
		extension: anExtension;
		description: aDescription;
		yourself
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> pytorch [

	"PyTorch format"
	^ self named: 'PyTorch' extension: 'pt' description: 'PyTorch model format'
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> safetensors [

	"Safetensors format"
	^ self named: 'SafeTensors' extension: 'safetensors' description: 'Safe serialization format for ML models'
]

{ #category : 'as yet unclassified' }
AIModelFormat class >> unknown [

	"Unknown format"
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
