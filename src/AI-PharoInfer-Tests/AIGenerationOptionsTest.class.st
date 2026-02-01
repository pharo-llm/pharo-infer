Class {
	#name : 'AIGenerationOptionsTest',
	#superclass : 'TestCase',
	#instVars : [
		'generator',
		'testModel'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'tests' }
AIGenerationOptionsTest >> testDefault [
	| options |

	options := AIGenerationOptions default.

	self assert: options temperature equals: 0.7.
	self assert: options topP equals: 0.9.
	self assert: options topK equals: 40.
	self assert: options maxTokens equals: 512.
	self deny: options stream
]

{ #category : 'tests' }
AIGenerationOptionsTest >> testSetOptions [
	| options |

	options := AIGenerationOptions new.
	options temperature: 0.5.
	options maxTokens: 1000.
	options stream: true.

	self assert: options temperature equals: 0.5.
	self assert: options maxTokens equals: 1000.
	self assert: options stream
]

{ #category : 'tests' }
AIGenerationOptionsTest >> testStopSequences [
	| options |

	options := AIGenerationOptions new.
	options stopSequences: #('\n' '###').

	self assert: options stopSequences size equals: 2.
	self assert: (options stopSequences includes: '\n').
	self assert: (options stopSequences includes: '###')
]
