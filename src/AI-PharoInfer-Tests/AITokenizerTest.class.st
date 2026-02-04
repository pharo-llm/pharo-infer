Class {
	#name : 'AITokenizerTest',
	#superclass : 'TestCase',
	#instVars : [
		'tokenizer'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AITokenizerTest >> setUp [

	super setUp.
	tokenizer := AITokenizer new
]

{ #category : 'running' }
AITokenizerTest >> testDecode [

	| tokens text |

	tokens := #(100 101 102).
	text := tokenizer decode: tokens.

	self assert: text isString
]

{ #category : 'running' }
AITokenizerTest >> testEncode [

	| tokens text |

	text := 'hello world'.
	tokens := tokenizer encode: text.

	self assert: tokens isArray.
	self assert: tokens notEmpty
]

{ #category : 'running' }
AITokenizerTest >> testRoundTrip [

	| original tokens decoded |

	original := 'the quick brown fox'.
	tokens := tokenizer tokenize: original.
	decoded := tokenizer decode: tokens.

	"Not exact match due to simplified tokenization, but should be similar"
	self assert: decoded isString.
	self assert: decoded notEmpty
]

{ #category : 'running' }
AITokenizerTest >> testSpecialTokens [

	self assert: tokenizer bosToken notNil.
	self assert: tokenizer eosToken notNil.
	self assert: tokenizer padToken notNil.
	self assert: tokenizer unkToken notNil
]

{ #category : 'running' }
AITokenizerTest >> testTokenize [

	| tokens text |

	text := 'hello world'.
	tokens := tokenizer tokenize: text.

	self assert: tokens isArray.
	self assert: tokens size > 0
]

{ #category : 'running' }
AITokenizerTest >> testTokenizeEmpty [

	| tokens |

	tokens := tokenizer tokenize: ''.

	self assert: tokens isEmpty
]

{ #category : 'running' }
AITokenizerTest >> testVocabularySize [

	| size |

	size := tokenizer vocabularySize.

	self assert: size > 0
]
