"
I hold configuration values for local inference.
I define model directories, threading, GPU offload layers, context
size and batch size used when instantiating a local llama.cpp backend.
"
Class {
	#name : 'AIInferenceConfig',
	#superclass : 'Object',
	#instVars : [
		'modelsDirectory',
		'cacheDirectory',
		'maxConcurrentInferences',
		'gpuLayers',
		'contextSize',
		'batchSize',
		'threads'
	],
	#classInstVars : [
		'default'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AIInferenceConfig class >> default [

	^ default ifNil: [ default := self new ]
]

{ #category : 'accessing' }
AIInferenceConfig class >> reset [

	<script>
	default := nil
]

{ #category : 'accessing' }
AIInferenceConfig >> batchSize [

	^ batchSize ifNil: [ 512 ]
]

{ #category : 'accessing' }
AIInferenceConfig >> batchSize: anObject [

	batchSize := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> cacheDirectory [

	^ cacheDirectory ifNil: [
		cacheDirectory := FileLocator home / 'pharo-models' / 'cache' ]
]

{ #category : 'accessing' }
AIInferenceConfig >> cacheDirectory: anObject [

	cacheDirectory := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> contextSize [

	^ contextSize ifNil: [ 2048 ]
]

{ #category : 'accessing' }
AIInferenceConfig >> contextSize: anObject [

	contextSize := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> gpuLayers [

	^ gpuLayers ifNil: [ 0 ]
]

{ #category : 'accessing' }
AIInferenceConfig >> gpuLayers: anObject [

	gpuLayers := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> initialize [

	super initialize.
	maxConcurrentInferences := 1.
	gpuLayers := 0
]

{ #category : 'accessing' }
AIInferenceConfig >> maxConcurrentInferences [

	^ maxConcurrentInferences
]

{ #category : 'accessing' }
AIInferenceConfig >> maxConcurrentInferences: anObject [

	maxConcurrentInferences := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> modelsDirectory [

	^ modelsDirectory ifNil: [
		modelsDirectory := FileLocator home / 'pharo-models' ]
]

{ #category : 'accessing' }
AIInferenceConfig >> modelsDirectory: anObject [

	modelsDirectory := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> threads [

	^ threads
]

{ #category : 'accessing' }
AIInferenceConfig >> threads: anObject [

	threads := anObject
]
