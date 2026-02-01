Class {
	#name : 'AILlamaCppBackend',
	#superclass : 'Object',
	#instVars : [
		'serverUrl',
		'client',
		'llamaCppPath'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'private' }
AILlamaCppBackend >> createClient [

	"Create HTTP client for llama.cpp server API"
	^ ZnClient new
		beOneShot;
		yourself
]

{ #category : 'private' }
AILlamaCppBackend >> generateEmbeddings: text model: aModel [

	"Generate embeddings using llama.cpp server"
	| response json |

	client := self createClient.
	client url: self serverUrl, '/embedding'.

	response := client
		contents: (STONJSON toString: (Dictionary newFromPairs: {
			'content'. text }));
		post.

	json := STONJSON fromString: response.

	^ json at: 'embedding'
]

{ #category : 'private' }
AILlamaCppBackend >> generateText: prompt model: aModel options: options [

	"Generate text using llama.cpp server"
	| response json |

	client := self createClient.
	client url: self serverUrl, '/completion'.

	response := client
		contents: (STONJSON toString: (Dictionary newFromPairs: {
			'prompt'. prompt.
			'stream'. false.
			'temperature'. options temperature.
			'top_p'. options topP.
			'top_k'. options topK.
			'n_predict'. options maxTokens }));
		post.

	json := STONJSON fromString: response.

	^ json at: 'content'
]

{ #category : 'private' }
AILlamaCppBackend >> initialize [

	super initialize.
	llamaCppPath := 'llama-server'
]

{ #category : 'private' }
AILlamaCppBackend >> llamaCppPath [

	^ llamaCppPath
]

{ #category : 'private' }
AILlamaCppBackend >> llamaCppPath: anObject [

	llamaCppPath := anObject
]

{ #category : 'private' }
AILlamaCppBackend >> loadModel: aModel [

	"Start llama.cpp server with the model"
	"In a full implementation, this would start the server process"
	"For now, assumes server is already running"
]

{ #category : 'private' }
AILlamaCppBackend >> serverUrl [

	^ serverUrl ifNil: [ serverUrl := 'http://localhost:8080' ]
]

{ #category : 'private' }
AILlamaCppBackend >> serverUrl: anObject [

	serverUrl := anObject
]

{ #category : 'private' }
AILlamaCppBackend >> streamText: prompt model: aModel options: options onToken: aBlock [

	"Stream text generation using llama.cpp server"
	| request |

	client := self createClient.
	client url: self serverUrl, '/completion'.
	client streaming: true.

	request := Dictionary newFromPairs: {
		'prompt'. prompt.
		'stream'. true.
		'temperature'. options temperature.
		'top_p'. options topP.
		'top_k'. options topK.
		'n_predict'. options maxTokens }.

	client
		contents: (STONJSON toString: request);
		streamingPost: [ :chunk |
			| json |
			json := STONJSON fromString: chunk.
			(json includesKey: 'content') ifTrue: [
				aBlock value: (json at: 'content') ].
			json at: 'stop' ifAbsent: [ false ] ]
]

{ #category : 'private' }
AILlamaCppBackend >> unloadModel: aModel [

	"Stop llama.cpp server (no-op for now)"
]
