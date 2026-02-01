Class {
	#name : 'AIInferenceConfig',
	#superclass : 'Object',
	#instVars : [
		'modelsDirectory',
		'cacheDirectory',
		'defaultBackend',
		'maxConcurrentInferences',
		'enableGPU',
		'gpuLayers',
		'contextSize',
		'batchSize'
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
AIInferenceConfig >> defaultBackend [

	^ defaultBackend ifNil: [ defaultBackend := #local ]
]

{ #category : 'accessing' }
AIInferenceConfig >> defaultBackend: anObject [

	defaultBackend := anObject
]

{ #category : 'accessing' }
AIInferenceConfig >> enableGPU [ 

	^ enableGPU ifNil: [ false ]
]

{ #category : 'accessing' }
AIInferenceConfig >> enableGPU: anObject [

	enableGPU := anObject
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
	enableGPU := false.
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
