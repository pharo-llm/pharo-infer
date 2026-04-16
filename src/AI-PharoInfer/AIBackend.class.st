"
""I am an abstract backend for AI inference.
Subclasses implement model loading, text generation, streaming, and embeddings generation.""
"
Class {
	#name : 'AIBackend',
	#superclass : 'Object',
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'operations' }
AIBackend >> generateEmbeddings: text model: aModel [
	"Generate embeddings for the given text. Subclasses should implement this."
	self subclassResponsibility
]

{ #category : 'operations' }
AIBackend >> generateText: prompt model: aModel options: options [ 
	"Generate text from a prompt. Subclasses should implement this."
	self subclassResponsibility
]

{ #category : 'operations' }
AIBackend >> loadModel: aModel [
	"Load a model into memory. Subclasses should implement this."
	self subclassResponsibility
]

{ #category : 'operations' }
AIBackend >> streamText: prompt model: aModel options: options onToken: aBlock [
	"Stream text generation, calling aBlock for each token. Subclasses should implement this."
	self subclassResponsibility
]

{ #category : 'operations' }
AIBackend >> unloadModel: aModel [
	"Unload a model from memory. Subclasses should implement this."
	self subclassResponsibility
]
