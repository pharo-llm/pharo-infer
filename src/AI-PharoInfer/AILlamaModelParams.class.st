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
		bool vocab_only
		bool use_mmap
		bool use_mlock
		bool check_tensors
	)
]

{ #category : 'accessing' }
AILlamaModelParams >> nGpuLayers [
	^ self int32AtOffset: OFFSET_N_GPU_LAYERS
]

{ #category : 'accessing' }
AILlamaModelParams >> nGpuLayers: anInteger [
	self int32AtOffset: OFFSET_N_GPU_LAYERS put: anInteger
]

{ #category : 'accessing' }
AILlamaModelParams >> useMmap [
	^ (self uint8AtOffset: OFFSET_USE_MMAP) ~= 0
]

{ #category : 'accessing' }
AILlamaModelParams >> useMmap: aBoolean [
	self uint8AtOffset: OFFSET_USE_MMAP put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
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
AILlamaModelParams >> vocabOnly [
	^ (self uint8AtOffset: OFFSET_VOCAB_ONLY) ~= 0
]

{ #category : 'accessing' }
AILlamaModelParams >> vocabOnly: aBoolean [
	self uint8AtOffset: OFFSET_VOCAB_ONLY put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
]
