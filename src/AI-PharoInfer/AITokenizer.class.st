Class {
	#name : 'AITokenizer',
	#superclass : 'Object',
	#instVars : [
		'vocabulary',
		'reverseVocabulary',
		'specialTokens',
		'bosToken',
		'eosToken',
		'padToken',
		'unkToken'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'accessing' }
AITokenizer >> bosToken [

	"Beginning of sequence token"
	^ bosToken ifNil: [ 1 ]
]

{ #category : 'accessing' }
AITokenizer >> bosToken: anObject [

	bosToken := anObject
]

{ #category : 'accessing' }
AITokenizer >> buildReverseVocabulary [

	"Build reverse vocabulary (ID -> string) from vocabulary (string -> ID)"

	reverseVocabulary := Dictionary new.
	vocabulary keysAndValuesDo: [ :string :id |
		reverseVocabulary at: id put: string ]
]

{ #category : 'accessing' }
AITokenizer >> decode: tokens [

	"Decode tokens back to text"
	| text |

	tokens isNumber ifTrue: [
		^ self decodeToken: tokens ].

	text := String streamContents: [ :stream |
		tokens do: [ :token |
			stream nextPutAll: (self decodeToken: token) ] ].

	^ text
]

{ #category : 'accessing' }
AITokenizer >> decodeToken: token [

	"Special tokens"
	token = self bosToken ifTrue: [ ^ '' ].
	token = self eosToken ifTrue: [ ^ '' ].
	token = self padToken ifTrue: [ ^ '' ].
	token = self unkToken ifTrue: [ ^ '<UNK>' ].

	"Look up in reverse vocabulary (ID -> string)"
	^ self reverseVocabulary
		at: token
		ifAbsent: [ '<UNK>' ]
]

{ #category : 'accessing' }
AITokenizer >> encode: text [

	"Encode text to token IDs"
	^ self tokenize: text
]

{ #category : 'accessing' }
AITokenizer >> eosToken [

	"End of sequence token"
	^ eosToken ifNil: [ 2 ]
]

{ #category : 'accessing' }
AITokenizer >> eosToken: anObject [

	eosToken := anObject
]

{ #category : 'accessing' }
AITokenizer >> initialize [

	super initialize.
	vocabulary := self initializeVocabulary.
	self buildReverseVocabulary.
	self updateSpecialTokens
]

{ #category : 'accessing' }
AITokenizer >> initializeSpecialTokens [

	"Initialize special tokens"
	^ Dictionary newFromPairs: {
		'<BOS>'. self bosToken.
		'<EOS>'. self eosToken.
		'<PAD>'. self padToken.
		'<UNK>'. self unkToken }
]

{ #category : 'accessing' }
AITokenizer >> initializeVocabulary [
    "Initialize a simple vocabulary (in reality, this would be loaded from model)"
    
    | vocab nextTokenId |
    
    vocab := Dictionary new.
    nextTokenId := 100.

    "Add some common words"
    'the a an and or but is are was were be been being have has had do does did will would could should may might must can to from in on at by for with about as into through during before after above below between among '
        splitOn: ' '
        do: [:word |
            word isEmpty ifFalse: [
                vocab at: word put: nextTokenId.
                nextTokenId := nextTokenId + 1
            ]
        ].

    "Add lowercase alphabet"
    $a asciiValue to: $z asciiValue do: [:i |
        | char |
        char := Character value: i.
        vocab at: char asString put: nextTokenId.
        nextTokenId := nextTokenId + 1
    ].

    "Add uppercase alphabet"
    $A asciiValue to: $Z asciiValue do: [:i |
        | char |
        char := Character value: i.
        vocab at: char asString put: nextTokenId.
        nextTokenId := nextTokenId + 1
    ].

    "Add digits"
    $0 asciiValue to: $9 asciiValue do: [:i |
        | char |
        char := Character value: i.
        vocab at: char asString put: nextTokenId.
        nextTokenId := nextTokenId + 1
    ].

    "Add punctuation"
    '.,!?;:-()[]{}' do: [:char |
        vocab at: char asString put: nextTokenId.
        nextTokenId := nextTokenId + 1
    ].

    "Add whitespace"
    vocab at: ' ' put: nextTokenId.
    nextTokenId := nextTokenId + 1.

    vocab at: String tab put: nextTokenId.
    nextTokenId := nextTokenId + 1.

    vocab at: String cr put: nextTokenId.
    nextTokenId := nextTokenId + 1.

    ^ vocab

]

{ #category : 'accessing' }
AITokenizer >> padToken [

	"Padding token"
	^ padToken ifNil: [ 0 ]
]

{ #category : 'accessing' }
AITokenizer >> padToken: anObject [

	padToken := anObject
]

{ #category : 'accessing' }
AITokenizer >> reverseVocabulary [

reverseVocabulary ifNil: [
		vocabulary ifNil: [ ^ Dictionary new ].
		self buildReverseVocabulary ].
	^ reverseVocabulary
]

{ #category : 'accessing' }
AITokenizer >> reverseVocabulary: anObject [

	reverseVocabulary := anObject
]

{ #category : 'accessing' }
AITokenizer >> specialTokens [

	^ specialTokens
]

{ #category : 'accessing' }
AITokenizer >> tokenize: text [

	"Tokenize text into token IDs (simplified word-level tokenization)"
	| tokens words |

	words := self tokenizeToWords: text.

	tokens := words collect: [ :word |
		self vocabulary
			at: word
			ifAbsent: [ self unkToken ] ].

	^ tokens asArray
]

{ #category : 'accessing' }
AITokenizer >> tokenizeToWords: text [

	"Simple word tokenization (in reality, would use BPE or WordPiece)"
	| words currentWord |

	words := OrderedCollection new.
	currentWord := WriteStream on: String new.

	text do: [ :char |
		(char isSeparator or: [ '.,!?;:-()[]{}''""' includes: char ]) ifTrue: [
			currentWord contents ifNotEmpty: [
				words add: currentWord contents.
				currentWord := WriteStream on: String new ].
			char isSeparator ifFalse: [
				words add: char asString ] ]
		ifFalse: [
			currentWord nextPut: char ] ].

	currentWord contents ifNotEmpty: [
		words add: currentWord contents ].

	^ words
]

{ #category : 'accessing' }
AITokenizer >> unkToken [

	"Unknown token"
	^ unkToken ifNil: [ 3 ]
]

{ #category : 'accessing' }
AITokenizer >> unkToken: anObject [

	unkToken := anObject
]

{ #category : 'accessing' }
AITokenizer >> updateSpecialTokens [

	"Update special tokens dictionary"

	specialTokens := self initializeSpecialTokens
]

{ #category : 'accessing' }
AITokenizer >> vocabulary [

	^ vocabulary
]

{ #category : 'accessing' }
AITokenizer >> vocabulary: anObject [

	vocabulary := anObject.
	self buildReverseVocabulary.
	self updateSpecialTokens
]

{ #category : 'accessing' }
AITokenizer >> vocabularySize [

	"For loaded vocabularies (e.g., from GGUF), special tokens are already included"
	vocabulary ifNil: [
		^ specialTokens values max + 1 ].
	^ vocabulary size
]
