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
		'OFFSET_ABORT_CALLBACK',
		'OFFSET_ABORT_CALLBACK_DATA',
		'OFFSET_ATTENTION_TYPE',
		'OFFSET_CB_EVAL',
		'OFFSET_CB_EVAL_USER_DATA',
		'OFFSET_DEFRAG_THOLD',
		'OFFSET_EMBEDDINGS',
		'OFFSET_FLASH_ATTN',
		'OFFSET_LOGITS_ALL',
		'OFFSET_NO_PERF',
		'OFFSET_N_BATCH',
		'OFFSET_N_CTX',
		'OFFSET_N_SEQ_MAX',
		'OFFSET_N_THREADS',
		'OFFSET_N_THREADS_BATCH',
		'OFFSET_N_UBATCH',
		'OFFSET_OFFLOAD_KQV',
		'OFFSET_POOLING_TYPE',
		'OFFSET_ROPE_FREQ_BASE',
		'OFFSET_ROPE_FREQ_SCALE',
		'OFFSET_ROPE_SCALING_TYPE',
		'OFFSET_TYPE_K',
		'OFFSET_TYPE_V',
		'OFFSET_YARN_ATTN_FACTOR',
		'OFFSET_YARN_BETA_FAST',
		'OFFSET_YARN_BETA_SLOW',
		'OFFSET_YARN_EXT_FACTOR',
		'OFFSET_YARN_ORIG_CTX'
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
		uint8 logits_all
		uint8 embeddings
		uint8 offload_kqv
		uint8 flash_attn
		uint8 no_perf
		void *abort_callback
		void *abort_callback_data
	)
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> abort_callback [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_ABORT_CALLBACK) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> abort_callback: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_ABORT_CALLBACK put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> abort_callback_data [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_ABORT_CALLBACK_DATA) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> abort_callback_data: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_ABORT_CALLBACK_DATA put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> attention_type [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_ATTENTION_TYPE
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> attention_type: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_ATTENTION_TYPE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> cb_eval [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_CB_EVAL) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> cb_eval: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_CB_EVAL put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> cb_eval_user_data [
	"This method was automatically generated"
	^ExternalData fromHandle: (handle pointerAt: OFFSET_CB_EVAL_USER_DATA) type: ExternalType void asPointerType
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> cb_eval_user_data: anObject [
	"This method was automatically generated"
	handle pointerAt: OFFSET_CB_EVAL_USER_DATA put: anObject getHandle.
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> defrag_thold [
	"This method was automatically generated"
	^handle floatAt: OFFSET_DEFRAG_THOLD
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> defrag_thold: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_DEFRAG_THOLD put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> embeddings [
	"This method was automatically generated"

	^ (handle unsignedByteAt: OFFSET_EMBEDDINGS) ~= 0
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> embeddings: aBoolean [

	handle unsignedByteAt: OFFSET_EMBEDDINGS put: (aBoolean
			 ifTrue: [ 1 ]
			 ifFalse: [ 0 ])
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> flash_attn [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_FLASH_ATTN
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> flash_attn: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_FLASH_ATTN put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> logits_all [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_LOGITS_ALL
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> logits_all: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_LOGITS_ALL put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nBatch [

	^ handle unsignedLongAt: OFFSET_N_BATCH
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nBatch: anInteger [

	handle unsignedLongAt: OFFSET_N_BATCH put: anInteger
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nCtx [

	^ handle unsignedLongAt: OFFSET_N_CTX
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nCtx: anInteger [

	handle unsignedLongAt: OFFSET_N_CTX put: anInteger
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nThreads [

	^ handle signedLongAt: OFFSET_N_THREADS
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nThreads: anInteger [

	handle signedLongAt: OFFSET_N_THREADS put: anInteger
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nThreadsBatch [
	^ handle signedLongAt: OFFSET_N_THREADS_BATCH
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> nThreadsBatch: anInteger [


	handle signedLongAt: OFFSET_N_THREADS_BATCH put: anInteger
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_batch [
	"This method was automatically generated"
	^handle unsignedLongAt: OFFSET_N_BATCH
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_batch: anObject [
	"This method was automatically generated"
	handle unsignedLongAt: OFFSET_N_BATCH put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_ctx [
	"This method was automatically generated"
	^handle unsignedLongAt: OFFSET_N_CTX
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_ctx: anObject [
	"This method was automatically generated"
	handle unsignedLongAt: OFFSET_N_CTX put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_seq_max [
	"This method was automatically generated"
	^handle unsignedLongAt: OFFSET_N_SEQ_MAX
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_seq_max: anObject [
	"This method was automatically generated"
	handle unsignedLongAt: OFFSET_N_SEQ_MAX put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_threads [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_N_THREADS
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_threads: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_N_THREADS put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_threads_batch [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_N_THREADS_BATCH
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_threads_batch: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_N_THREADS_BATCH put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_ubatch [
	"This method was automatically generated"
	^handle unsignedLongAt: OFFSET_N_UBATCH
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> n_ubatch: anObject [
	"This method was automatically generated"
	handle unsignedLongAt: OFFSET_N_UBATCH put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> no_perf [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_NO_PERF
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> no_perf: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_NO_PERF put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> offload_kqv [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_OFFLOAD_KQV
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> offload_kqv: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_OFFLOAD_KQV put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> pooling_type [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_POOLING_TYPE
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> pooling_type: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_POOLING_TYPE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_freq_base [
	"This method was automatically generated"
	^handle floatAt: OFFSET_ROPE_FREQ_BASE
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_freq_base: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_ROPE_FREQ_BASE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_freq_scale [
	"This method was automatically generated"
	^handle floatAt: OFFSET_ROPE_FREQ_SCALE
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_freq_scale: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_ROPE_FREQ_SCALE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_scaling_type [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_ROPE_SCALING_TYPE
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> rope_scaling_type: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_ROPE_SCALING_TYPE put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> type_k [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_TYPE_K
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> type_k: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_TYPE_K put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> type_v [
	"This method was automatically generated"
	^handle signedLongAt: OFFSET_TYPE_V
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> type_v: anObject [
	"This method was automatically generated"
	handle signedLongAt: OFFSET_TYPE_V put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_attn_factor [
	"This method was automatically generated"
	^handle floatAt: OFFSET_YARN_ATTN_FACTOR
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_attn_factor: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_YARN_ATTN_FACTOR put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_beta_fast [
	"This method was automatically generated"
	^handle floatAt: OFFSET_YARN_BETA_FAST
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_beta_fast: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_YARN_BETA_FAST put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_beta_slow [
	"This method was automatically generated"
	^handle floatAt: OFFSET_YARN_BETA_SLOW
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_beta_slow: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_YARN_BETA_SLOW put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_ext_factor [
	"This method was automatically generated"
	^handle floatAt: OFFSET_YARN_EXT_FACTOR
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_ext_factor: anObject [
	"This method was automatically generated"
	handle floatAt: OFFSET_YARN_EXT_FACTOR put: anObject
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_orig_ctx [
	"This method was automatically generated"
	^handle unsignedLongAt: OFFSET_YARN_ORIG_CTX
]

{ #category : 'accessing - structure variables' }
AILlamaContextParams >> yarn_orig_ctx: anObject [
	"This method was automatically generated"
	handle unsignedLongAt: OFFSET_YARN_ORIG_CTX put: anObject
]
