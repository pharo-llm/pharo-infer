Class {
	#name : 'AIModelTest',
	#superclass : 'TestCase',
	#instVars : [
		'manager'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'tests' }
AIModelTest >> testCreateModel [

	| model |

	model := AIModel named: 'test-model'.

	self assert: model name equals: 'test-model'.
	self deny: model isLoaded
]

{ #category : 'tests' }
AIModelTest >> testFromFile [

	| model file |

	file := FileLocator temp / 'test-model.gguf'.
	file ensureCreateFile.

	[ model := AIModel fromFile: file.

	self assert: model name equals: 'test-model'.
	self assert: model path equals: file.
	self assert: model format name equals: 'GGUF' ]
		ensure: [ file ensureDelete ]
]

{ #category : 'tests' }
AIModelTest >> testLoadAndUnload [

	| model backend |

	backend := AILocalBackend new.
	model := AIModel named: 'test-model'.
	model backend: backend.
	model path: (FileLocator temp / 'test.gguf').
	model path ensureCreateFile.

	[ model load.
	self assert: model isLoaded.

	model unload.
	self deny: model isLoaded ]
		ensure: [ model path ensureDelete ]
]

{ #category : 'tests' }
AIModelTest >> testModelMetadata [

	| model |

	model := AIModel named: 'test-model'.
	model metadata at: 'author' put: 'Test Author'.
	model metadata at: 'version' put: '1.0'.

	self assert: (model metadata at: 'author') equals: 'Test Author'.
	self assert: (model metadata at: 'version') equals: '1.0'
]

{ #category : 'tests' }
AIModelTest >> testPrintOn [

	| model printString |

	model := AIModel named: 'test-model'.
	printString := model printString.

	self assert: (printString includesSubstring: 'test-model').
	self assert: (printString includesSubstring: 'unloaded')
]
