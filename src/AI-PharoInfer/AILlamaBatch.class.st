"
I mirror the llama.cpp C struct 'llama_batch'. Instances are created by
llama_batch_get_one() and consumed by llama_decode().

This legacy mirror is kept for source compatibility. The production
backend no longer passes this struct through UFFI; the native
PharoInfer shim creates llama_batch values inside C.
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
