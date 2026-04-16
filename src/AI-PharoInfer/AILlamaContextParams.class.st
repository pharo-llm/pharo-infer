"
I mirror the llama.cpp C struct 'llama_context_params' (by-value
parameter to llama_init_from_model). Layout aligned with the
b4000+ series.

Fetch defaults via AILlamaLibrary>>llamaContextDefaultParams, set
#nCtx:, #nBatch:, #nThreads:, #embeddings: as needed, then call
#llamaInitFromModel:params:.
"
Class {
	#name : 'AILlamaContextParams',
	#superclass : 'FFIExternalStructure',
	#classVars : [
		'OFFSET_EMBEDDINGS',
		'OFFSET_N_BATCH',
		'OFFSET_N_CTX',
		'OFFSET_N_THREADS',
		'OFFSET_N_THREADS_BATCH',
		'OFFSET_OFFLOAD_KQV',
		'OFFSET_FLASH_ATTN'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'field definition' }
AILlamaContextParams class >> fieldsDesc [

	^ #(
		uint32 n_ctx
		uint32 n_batch
		uint32 n_ubatch
		uint32 n_seq_max
		int32 n_threads
		int32 n_threads_batch
		int32 rope_scaling_type
		int32 pooling_type
		int32 attention_type
		float32 rope_freq_base
		float32 rope_freq_scale
		float32 yarn_ext_factor
		float32 yarn_attn_factor
		float32 yarn_beta_fast
		float32 yarn_beta_slow
		uint32 yarn_orig_ctx
		float32 defrag_thold
		void *cb_eval
		void *cb_eval_user_data
		int32 type_k
		int32 type_v
		bool logits_all
		bool embeddings
		bool offload_kqv
		bool flash_attn
		bool no_perf
		void *abort_callback
		void *abort_callback_data
	)
]

{ #category : 'accessing' }
AILlamaContextParams >> embeddings [
	^ (self uint8AtOffset: OFFSET_EMBEDDINGS) ~= 0
]

{ #category : 'accessing' }
AILlamaContextParams >> embeddings: aBoolean [
	self uint8AtOffset: OFFSET_EMBEDDINGS put: (aBoolean ifTrue: [ 1 ] ifFalse: [ 0 ])
]

{ #category : 'accessing' }
AILlamaContextParams >> nBatch [
	^ self uint32AtOffset: OFFSET_N_BATCH
]

{ #category : 'accessing' }
AILlamaContextParams >> nBatch: anInteger [
	self uint32AtOffset: OFFSET_N_BATCH put: anInteger
]

{ #category : 'accessing' }
AILlamaContextParams >> nCtx [
	^ self uint32AtOffset: OFFSET_N_CTX
]

{ #category : 'accessing' }
AILlamaContextParams >> nCtx: anInteger [
	self uint32AtOffset: OFFSET_N_CTX put: anInteger
]

{ #category : 'accessing' }
AILlamaContextParams >> nThreads [
	^ self int32AtOffset: OFFSET_N_THREADS
]

{ #category : 'accessing' }
AILlamaContextParams >> nThreads: anInteger [
	self int32AtOffset: OFFSET_N_THREADS put: anInteger
]

{ #category : 'accessing' }
AILlamaContextParams >> nThreadsBatch [
	^ self int32AtOffset: OFFSET_N_THREADS_BATCH
]

{ #category : 'accessing' }
AILlamaContextParams >> nThreadsBatch: anInteger [
	self int32AtOffset: OFFSET_N_THREADS_BATCH put: anInteger
]
