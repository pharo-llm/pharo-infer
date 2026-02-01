Class {
	#name : 'AIConfigTest',
	#superclass : 'TestCase',
	#instVars : [
		'config'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIConfigTest >> setUp [

	super setUp.
	config := AIInferenceConfig new
]

{ #category : 'running' }
AIConfigTest >> testBatchSize [

	self assert: config batchSize equals: 512.

	config batchSize: 1024.
	self assert: config batchSize equals: 1024
]

{ #category : 'running' }
AIConfigTest >> testCacheDirectory [
	| cacheDir |

	cacheDir := config cacheDirectory.

	self assert: cacheDir notNil.
	self assert: (cacheDir basename = 'cache')
]

{ #category : 'running' }
AIConfigTest >> testContextSize [
	self assert: config contextSize equals: 2048.

	config contextSize: 4096.
	self assert: config contextSize equals: 4096
]

{ #category : 'running' }
AIConfigTest >> testDefaultBackend [
	self assert: config defaultBackend equals: #local.

	config defaultBackend: #ollama.
	self assert: config defaultBackend equals: #ollama
]

{ #category : 'running' }
AIConfigTest >> testGPUSettings [
	self deny: config enableGPU.
	self assert: config gpuLayers equals: 0.

	config enableGPU: true.
	config gpuLayers: 32.

	self assert: config enableGPU.
	self assert: config gpuLayers equals: 32
]

{ #category : 'running' }
AIConfigTest >> testMaxConcurrentInferences [

	self assert: config maxConcurrentInferences equals: 1.

	config maxConcurrentInferences: 4.
	self assert: config maxConcurrentInferences equals: 4
]

{ #category : 'running' }
AIConfigTest >> testModelsDirectory [
	| modelsDir |

	modelsDir := config modelsDirectory.

	self assert: modelsDir notNil.
	self assert: (modelsDir basename = 'pharo-models')
]
