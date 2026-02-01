Class {
	#name : 'AIModelManager',
	#superclass : 'Object',
	#instVars : [
		'models',
		'modelsDirectory',
		'currentBackend'
	],
	#classInstVars : [
		'default'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AIModelManager class >> default [

	^ default ifNil: [ default := self new ]
]

{ #category : 'accessing' }
AIModelManager class >> reset [

	default := nil
]

{ #category : 'as yet unclassified' }
AIModelManager >> allModels [

	"Return all registered models"
	^ models values
]

{ #category : 'as yet unclassified' }
AIModelManager >> currentBackend [

	^ currentBackend ifNil: [ currentBackend := AILocalBackend new ]
]

{ #category : 'as yet unclassified' }
AIModelManager >> currentBackend: aBackend [

	currentBackend := aBackend
]

{ #category : 'as yet unclassified' }
AIModelManager >> discoverModelsInDirectory: aFileReference [

	"Discover and register all models in the given directory"
	| discovered |
	discovered := OrderedCollection new.

	aFileReference exists ifFalse: [ ^ discovered ].

	aFileReference allChildrenMatching: '*.gguf' do: [ :file |
		| model |
		model := AIModel fromFile: file.
		model backend: self currentBackend.
		self registerModel: model.
		discovered add: model ].

	aFileReference allChildrenMatching: '*.safetensors' do: [ :file |
		| model |
		model := AIModel fromFile: file.
		model backend: self currentBackend.
		self registerModel: model.
		discovered add: model ].

	^ discovered
]

{ #category : 'as yet unclassified' }
AIModelManager >> initialize [

	super initialize.
	models := Dictionary new.
	modelsDirectory := FileLocator home / 'pharo-models'
]

{ #category : 'as yet unclassified' }
AIModelManager >> listModels [

	"Return a collection of all model names"
	^ models keys
]

{ #category : 'as yet unclassified' }
AIModelManager >> loadModel: modelNameOrPath [

	"Load a model by name or path"
	| model |

	"Try to find by name first"
	model := models at: modelNameOrPath ifAbsent: [ nil ].

	"If not found, try to load from file"
	model ifNil: [
		| fileRef |
		fileRef := modelNameOrPath asFileReference.
		fileRef exists ifTrue: [
			model := AIModel fromFile: fileRef.
			model backend: self currentBackend.
			self registerModel: model ] ].

	model ifNil: [
		Error signal: 'Model not found: ', modelNameOrPath asString ].

	model load.
	^ model
]

{ #category : 'as yet unclassified' }
AIModelManager >> modelNamed: aString [

	"Get a model by name"
	^ models at: aString ifAbsent: [ nil ]
]

{ #category : 'as yet unclassified' }
AIModelManager >> models [

	^ models
]

{ #category : 'as yet unclassified' }
AIModelManager >> modelsDirectory [

	^ modelsDirectory
]

{ #category : 'as yet unclassified' }
AIModelManager >> modelsDirectory: aFileReference [

	modelsDirectory := aFileReference
]

{ #category : 'as yet unclassified' }
AIModelManager >> registerModel: aModel [

	"Register a model with the manager"
	models at: aModel name put: aModel
]

{ #category : 'as yet unclassified' }
AIModelManager >> unloadAll [

	"Unload all loaded models"
	models values do: [ :model |
		model isLoaded ifTrue: [ model unload ] ]
]

{ #category : 'as yet unclassified' }
AIModelManager >> unloadModel: modelName [

	"Unload a model by name"
	| model |
	model := models at: modelName ifAbsent: [
		Error signal: 'Model not found: ', modelName ].
	model unload
]

{ #category : 'as yet unclassified' }
AIModelManager >> unregisterModel: modelName [

	"Unregister a model"
	| model |
	model := models at: modelName ifAbsent: [ ^ self ].
	model isLoaded ifTrue: [ model unload ].
	models removeKey: modelName
]
