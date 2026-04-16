"
I mirror the llama.cpp C struct 'llama_sampler_chain_params'. A single
bool field (no_perf) in current releases.
"
Class {
	#name : 'AILlamaSamplerChainParams',
	#superclass : 'FFIExternalStructure',
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'field definition' }
AILlamaSamplerChainParams class >> fieldsDesc [

	^ #( bool no_perf )
]
