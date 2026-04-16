Class {
	#name : 'AIModelFormatTest',
	#superclass : 'TestCase',
	#category : 'AI-PharoInfer-Tests-Tests',
	#package : 'AI-PharoInfer-Tests',
	#tag : 'Tests'
}

{ #category : 'tests' }
AIModelFormatTest >> testDetectFromFile [
	| file format |

	file := FileLocator temp / 'test-model.gguf'.
	file ensureCreateFile.

	[ format := AIModelFormat detectFromFile: file.
	self assert: format name equals: 'GGUF' ]
		ensure: [ file ensureDelete ]
]

{ #category : 'tests' }
AIModelFormatTest >> testDetectFromFileUnknown [

	| file format |

	file := FileLocator temp / 'test-model.unknown'.
	file ensureCreateFile.

	[ format := AIModelFormat detectFromFile: file.
	self assert: format name equals: 'Unknown' ]
		ensure: [ file ensureDelete ]
]

{ #category : 'tests' }
AIModelFormatTest >> testFormats [

	| formats |

	formats := AIModelFormat formats.

	self assert: formats notEmpty.
	self assert: (formats anySatisfy: [ :f | f name = 'GGUF' ])
]

{ #category : 'tests' }
AIModelFormatTest >> testGGUF [

	| format |

	format := AIModelFormat gguf.

	self assert: format name equals: 'GGUF'.
	self assert: format extension equals: 'gguf'
]
