Class {
	#name : 'AIGGUFParser',
	#superclass : 'Object',
	#instVars : [
		'stream',
		'version',
		'metadata',
		'tensorCount',
		'metadataCount'
	],
	#category : 'AI-PharoInfer',
	#package : 'AI-PharoInfer'
}

{ #category : 'parsing' }
AIGGUFParser class >> parseFile: aFileReference [
	"Parse a GGUF file and return the parser instance"

	^ self new parseFile: aFileReference
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> extractTokenizerData [

	"Extract tokenizer data from metadata"
	| tokenizerData tokens |

	tokenizerData := Dictionary new.

	"Extract tokens array"
	tokens := metadata at: 'tokenizer.ggml.tokens' ifAbsent: [ nil ].
	tokens ifNotNil: [
		tokenizerData at: 'tokens' put: tokens ].

	"Extract special token IDs"
	#('bos_token_id' 'eos_token_id' 'pad_token_id' 'unk_token_id') do: [ :key |
		| metaKey value |
		metaKey := 'tokenizer.ggml.' , key.
		value := metadata at: metaKey ifAbsent: [ nil ].
		value ifNotNil: [
			tokenizerData at: key put: value ] ].

	^ tokenizerData
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> metadata [

	^ metadata
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> parseFile: aFileReference [

	"Parse a GGUF file"

	aFileReference asFileReference exists ifFalse: [
		Error signal: 'File not found: ', aFileReference fullName ].

	stream := aFileReference asFileReference binaryReadStream.

	[
		self parseHeader.
		self parseMetadata.
	] ensure: [ stream close ].

	^ self
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> parseHeader [

	"Parse GGUF file header"
	| magic |

	"Read magic number (4 bytes)"
	magic := ByteArray new: 4.
	1 to: 4 do: [ :i | magic at: i put: stream next ].
	(magic utf8Decoded = 'GGUF') ifFalse: [
		Error signal: 'Invalid GGUF file: wrong magic number' ].

	"Read version (uint32, little-endian)"
	version := self readUInt32.

	"Read tensor count (uint64, little-endian)"
	tensorCount := self readUInt64.

	"Read metadata count (uint64, little-endian)"
	metadataCount := self readUInt64.
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> parseMetadata [
	"Parse all metadata key-value pairs"

	metadata := Dictionary new.
	metadataCount timesRepeat: [
			| key valueType value |
			"Read key"
			key := self readKeyString. "Read value type (uint32)"
			valueType := self readUInt32. "Read value based on type"
			value := self readValue: valueType.

			metadata at: key put: value ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readArray [
	"Read an array value"

	| arrayType arrayLength values |
	"Read array value type (uint32)"
	arrayType := self readUInt32. "Read array length (uint64)"
	arrayLength := self readUInt64.
	arrayLength > self remainingBytes ifTrue: [ Error signal: 'Invalid GGUF array length: ' , arrayLength asString ]. "Read array values"
	values := OrderedCollection new: (self readArrayLength: arrayLength).
	arrayLength timesRepeat: [ values add: (self readValue: arrayType) ].

	^ values asArray
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readArrayLength: arrayLength [
	"Validate the array length can be allocated"

	arrayLength > SmallInteger maxVal ifTrue: [
		Error signal: 'GGUF array too large: ', arrayLength asString ].

	^ arrayLength asInteger
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readFloat32 [

	"Read a 32-bit float (little-endian)"
	| bytes value |

	bytes := ByteArray new: 4.
	1 to: 4 do: [ :i | bytes at: i put: stream next ].

	"Convert little-endian bytes to integer"
	value := (bytes at: 4) << 24 + ((bytes at: 3) << 16) + ((bytes at: 2) << 8) + (bytes at: 1).

	"Convert IEEE 754 binary32 to float"
	^ Float fromIEEE32Bit: value
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readFloat64 [

	"Read a 64-bit float (little-endian)"
	| bytes value |

	bytes := ByteArray new: 8.
	1 to: 8 do: [ :i | bytes at: i put: stream next ].

	"Convert little-endian bytes to 64-bit value"
	value := ((bytes at: 8) asInteger << 56) + ((bytes at: 7) asInteger << 48) +
		((bytes at: 6) asInteger << 40) + ((bytes at: 5) asInteger << 32) +
		((bytes at: 4) asInteger << 24) + ((bytes at: 3) asInteger << 16) +
		((bytes at: 2) asInteger << 8) + (bytes at: 1) asInteger.

	"Convert IEEE 754 binary64 to float"
	^ Float fromIEEE64Bit: value
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readInt16 [
	"Read a signed 16-bit integer (little-endian)"
	| value |

	value := self readUInt16.

	"Handle sign"
	^ value > 16r7FFF
		ifTrue: [ value - 16r10000 ]
		ifFalse: [ value ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readInt32 [
	"Read a signed 32-bit integer (little-endian)"
	| value |

	value := self readUInt32.

	"Handle sign"
	^ value > 16r7FFFFFFF
		ifTrue: [ value - 16r100000000 ]
		ifFalse: [ value ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readInt64 [

	"Read a signed 64-bit integer (little-endian)"
	| value |

	value := self readUInt64.

	"Handle sign for very large numbers"
	^ value > 16r7FFFFFFFFFFFFFFF
		ifTrue: [ value - 16r10000000000000000 ]
		ifFalse: [ value ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readInt8 [

	"Read a signed 8-bit integer"
	| byte |

	byte := stream next.

	"Handle sign"
	^ byte > 16r7F
		ifTrue: [ byte - 16r100 ]
		ifFalse: [ byte ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readKeyString [

	"Read a metadata key string (uint32 length + UTF-8 bytes)"
	| length bytes |

	"Read string length (uint32)"
	length := self readUInt32.

	(length > self remainingBytes) ifTrue: [
		Error signal: 'Invalid GGUF key length: ', length asString ].

	bytes := stream next: length.

	"Convert UTF-8 bytes to string"
	^ bytes utf8Decoded
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readString [

	"Read a string value (uint64 length + UTF-8 bytes)"
	| length bytes |

	"Read string length (uint64)"
	length := self readUInt64.
	
	(length > self remainingBytes) ifTrue: [
		Error signal: 'Invalid GGUF string length: ', length asString ].

	bytes := stream next: length.
	^ bytes utf8Decoded
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt16 [

	"Read an unsigned 16-bit integer (little-endian)"
	| b1 b2 |

	b1 := stream next.
	b2 := stream next.
	^ (b2 << 8) + b1
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt32 [

	"Read an unsigned 32-bit integer (little-endian)"
	| b1 b2 b3 b4 |

	b1 := stream next.
	b2 := stream next.
	b3 := stream next.
	b4 := stream next.
	^ (b4 << 24) + (b3 << 16) + (b2 << 8) + b1
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt64 [

	"Read an unsigned 64-bit integer (little-endian)"
	| b1 b2 b3 b4 b5 b6 b7 b8 |

	b1 := stream next.
	b2 := stream next.
	b3 := stream next.
	b4 := stream next.
	b5 := stream next.
	b6 := stream next.
	b7 := stream next.
	b8 := stream next.

	^ ((b8 asInteger << 56) + (b7 asInteger << 48) +
		(b6 asInteger << 40) + (b5 asInteger << 32) +
		(b4 asInteger << 24) + (b3 asInteger << 16) +
		(b2 asInteger << 8) + b1 asInteger)
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt8 [

	"Read an unsigned 8-bit integer"

	^ stream next
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readValue: valueType [

	"Read a value based on its type"

	"GGUF metadata value types:
	0 = UINT8, 1 = INT8, 2 = UINT16, 3 = INT16,
	4 = UINT32, 5 = INT32, 6 = UINT64, 7 = INT64,
	8 = FLOAT32, 9 = FLOAT64, 10 = BOOL, 11 = STRING, 12 = ARRAY"

	valueType = 0 ifTrue: [ ^ self readUInt8 ].
	valueType = 1 ifTrue: [ ^ self readInt8 ].
	valueType = 2 ifTrue: [ ^ self readUInt16 ].
	valueType = 3 ifTrue: [ ^ self readInt16 ].
	valueType = 4 ifTrue: [ ^ self readUInt32 ].
	valueType = 5 ifTrue: [ ^ self readInt32 ].
	valueType = 6 ifTrue: [ ^ self readUInt64 ].
	valueType = 7 ifTrue: [ ^ self readInt64 ].
	valueType = 8 ifTrue: [ ^ self readFloat32 ].
	valueType = 9 ifTrue: [ ^ self readFloat64 ].
	valueType = 10 ifTrue: [ ^ self readUInt8 ~= 0 ].  "BOOL"
	valueType = 11 ifTrue: [ ^ self readString ].
	valueType = 12 ifTrue: [ ^ self readArray ].

	Error signal: 'Unknown GGUF value type: ', valueType asString
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> remainingBytes [

	"Return remaining bytes in the stream"
	^ stream size - stream position
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> tokenizerData [

	"Extract and return tokenizer data from metadata"

	^ self extractTokenizerData
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> version [

	^ version
]
