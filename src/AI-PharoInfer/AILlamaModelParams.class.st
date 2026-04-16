"
I mirror the llama.cpp C struct 'llama_model_params' (by-value parameter
to llama_model_load_from_file). Layout is aligned with the
b4000+ series of llama.cpp.

Fetch a pre-filled instance via AILlamaLibrary>>llamaModelDefaultParams,
mutate the fields you care about (typically #nGpuLayers: or #useMmap:),
then pass it to #llamaModelLoadFromFile:params:.
"
Class {
	#name : 'AILlamaModelParams',
	#superclass : 'FFIExternalStructure',
	#classVars : [
		'OFFSET_CHECK_TENSORS',
		'OFFSET_DEVICES',
		'OFFSET_KV_OVERRIDES',
		'OFFSET_MAIN_GPU',
		'OFFSET_N_GPU_LAYERS',
		'OFFSET_PROGRESS_CALLBACK',
		'OFFSET_PROGRESS_CALLBACK_USER_DATA',
		'OFFSET_SPLIT_MODE',
		'OFFSET_TENSOR_BUFT_OVERRIDES',
		'OFFSET_TENSOR_SPLIT',
		'OFFSET_USE_MLOCK',
		'OFFSET_USE_MMAP',
		'OFFSET_VOCAB_ONLY'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'field definition' }
AILlamaModelParams class >> fieldsDesc [

		^ #(
		void *devices
		void *tensor_buft_overrides
		int32 n_gpu_layers
		int32 split_mode
		int32 main_gpu
		void *tensor_split
		void *progress_callback
		void *progress_callback_user_data
		void *kv_overrides
		uint8 vocab_only
		uint8 use_mmap
		uint8 use_mlock
		uint8 check_tensors
	)
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> check_tensors [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_CHECK_TENSORS
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> check_tensors: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_CHECK_TENSORS put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> devices [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_DEVICES) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> devices: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_DEVICES put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> kv_overrides [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_KV_OVERRIDES) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> kv_overrides: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_KV_OVERRIDES put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> main_gpu [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_MAIN_GPU
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> main_gpu: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_MAIN_GPU put: anObject
]

{ #category : 'accessing' }
AILlamaModelParams >> nGpuLayers [
	^ self int32AtOffset: OFFSET_N_GPU_LAYERS
]

{ #category : 'accessing' }
AILlamaModelParams >> nGpuLayers: anInteger [
	self int32AtOffset: OFFSET_N_GPU_LAYERS put: anInteger
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> n_gpu_layers [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_N_GPU_LAYERS
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> n_gpu_layers: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_N_GPU_LAYERS put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> progress_callback [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_PROGRESS_CALLBACK) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> progress_callback: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_PROGRESS_CALLBACK put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> progress_callback_user_data [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_PROGRESS_CALLBACK_USER_DATA) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> progress_callback_user_data: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_PROGRESS_CALLBACK_USER_DATA put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> split_mode [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_SPLIT_MODE
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> split_mode: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_SPLIT_MODE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> tensor_buft_overrides [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_TENSOR_BUFT_OVERRIDES) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> tensor_buft_overrides: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_TENSOR_BUFT_OVERRIDES put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> tensor_split [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_TENSOR_SPLIT) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> tensor_split: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_TENSOR_SPLIT put: anObject getHandle.
]

{ #category : 'accessing' }
AILlamaModelParams >> useMlock [
	^ (self uint8AtOffset: OFFSET_USE_MLOCK) ~= 0
]

{ #category : 'accessing' }
AILlamaModelParams >> useMlock: aBoolean [
	self uint8AtOffset: OFFSET_USE_MLOCK put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
]

{ #category : 'accessing' }
AILlamaModelParams >> useMmap [
	^ (self uint8AtOffset: OFFSET_USE_MMAP) ~= 0
]

{ #category : 'accessing' }
AILlamaModelParams >> useMmap: aBoolean [
	self uint8AtOffset: OFFSET_USE_MMAP put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> use_mlock [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_USE_MLOCK
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> use_mlock: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_USE_MLOCK put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> use_mmap [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_USE_MMAP
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> use_mmap: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_USE_MMAP put: anObject
]

{ #category : 'accessing' }
AILlamaModelParams >> vocabOnly [
	^ (self uint8AtOffset: OFFSET_VOCAB_ONLY) ~= 0
]

{ #category : 'accessing' }
AILlamaModelParams >> vocabOnly: aBoolean [
	self uint8AtOffset: OFFSET_VOCAB_ONLY put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> vocab_only [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_VOCAB_ONLY
]

{ #category : 'accessing - structure variables' }
AILlamaModelParams >> vocab_only: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_VOCAB_ONLY put: anObject
]
