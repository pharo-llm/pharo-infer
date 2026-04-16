"
I mirror the llama.cpp C struct 'llama_batch'. Instances are created by
llama_batch_get_one() and consumed by llama_decode().

For simple use cases we never touch the fields directly: we just obtain
a batch from #llamaBatchGetOne:nTokens: and hand it back to
#llamaDecode:batch:.
"
Class {
	#name : 'AILlamaBatch',
	#superclass : 'FFIExternalStructure',
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'field definition' }
AILlamaBatch class >> fieldsDesc [

	^ #(
		int32 n_tokens
		void *token
		void *embd
		void *pos
		void *n_seq_id
		void *seq_id
		void *logits
	)
]
