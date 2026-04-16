"
I mirror the llama.cpp C struct 'llama_sampler_chain_params'. A single
bool field (no_perf) in current releases.
"
Class {
	#name : 'AILlamaSamplerChainParams',
	#superclass : 'FFIExternalStructure',
	#classVars : [
		'OFFSET_NO_PERF'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'field definition' }
AILlamaSamplerChainParams class >> fieldsDesc [

^ #( uint8 no_perf )
]

{ #category : 'accessing - structure variables' }
AILlamaSamplerChainParams >> no_perf [
	"This method was automatically generated"
	^handle unsignedByteAt: OFFSET_NO_PERF
]

{ #category : 'accessing - structure variables' }
AILlamaSamplerChainParams >> no_perf: anObject [
	"This method was automatically generated"
	handle unsignedByteAt: OFFSET_NO_PERF put: anObject
]
