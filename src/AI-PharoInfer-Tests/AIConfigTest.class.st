Class {
	#name : 'AIConfigTest',
	#superclass : 'TestCase',
	#instVars : [
		'config'
	],
	#category : 'AI-PharoInfer-Tests-Tests',
	#package : 'AI-PharoInfer-Tests',
	#tag : 'Tests'
}

{ #category : 'running' }
AIConfigTest >> setUp [

	super setUp.
	config := AIInferenceConfig new
]

{ #category : 'tests' }
AIConfigTest >> testBatchSize [

	self assert: config batchSize equals: 512.

	config batchSize: 1024.
	self assert: config batchSize equals: 1024
]

{ #category : 'tests' }
AIConfigTest >> testCacheDirectory [

	| cacheDir |
	cacheDir := config cacheDirectory.

	self assert: cacheDir notNil.
	self assert: cacheDir basename equals: 'cache'
]

{ #category : 'tests' }
AIConfigTest >> testContextSize [

	self assert: config contextSize equals: 2048.

	config contextSize: 4096.
	self assert: config contextSize equals: 4096
]

{ #category : 'tests' }
AIConfigTest >> testGpuLayers [

	self assert: config gpuLayers equals: 0.

	config gpuLayers: 32.
	self assert: config gpuLayers equals: 32
]

{ #category : 'tests' }
AIConfigTest >> testMaxConcurrentInferences [

	self assert: config maxConcurrentInferences equals: 1.

	config maxConcurrentInferences: 4.
	self assert: config maxConcurrentInferences equals: 4
]

{ #category : 'tests' }
AIConfigTest >> testModelsDirectory [

	| modelsDir |
	modelsDir := config modelsDirectory.

	self assert: modelsDir notNil.
	self assert: modelsDir basename equals: 'pharo-models'
]

{ #category : 'tests' }
AIConfigTest >> testThreads [

	self assert: config threads isNil.

	config threads: 8.
	self assert: config threads equals: 8
]
