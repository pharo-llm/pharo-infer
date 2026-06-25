Class {
	#name : 'AIInferenceEngineTest',
	#superclass : 'TestCase',
	#instVars : [
		'manager',
		'engine'
	],
	#category : 'AI-PharoInfer-Tests-Tests',
	#package : 'AI-PharoInfer-Tests',
	#tag : 'Tests'
}

{ #category : 'running' }
AIInferenceEngineTest >> setUp [

	super setUp.
	manager := AIModelManager new.
	engine := AIInferenceEngine new.
	engine modelManager: manager
]

{ #category : 'tests' }
AIInferenceEngineTest >> testCompleteUsesModelBackend [

	| file model response |
	file := FileLocator temp / 'engine-backend-test.gguf'.
	file ensureCreateFile.

	[
		model := AIModel fromFile: file.
		model backend: AIMockBackend new.
		manager registerModel: model.

		response := engine complete: 'hello' model: model name.

		self assert: response equals: 'Mock response for: hello'.
		self assert: model isLoaded ]
		ensure: [ file ensureDelete ]
]
