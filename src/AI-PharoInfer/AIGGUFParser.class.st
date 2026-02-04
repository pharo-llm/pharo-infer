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
AIGGUFParser >> decodeUtf8Bytes: bytes [

	"Decode UTF-8 bytes while tolerating invalid sequences."
	^ [ bytes utf8Decoded ]
		on: ZnCharacterEncodingError
		do: [ :error | bytes asString ]
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
| fileRef |
fileRef := aFileReference asFileReference.

fileRef exists ifFalse: [
	Error signal: 'File not found: ', fileRef fullName ].

stream := fileRef binaryReadStream.

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
magic := stream next: 4.
(magic isNil or: [ magic size < 4 ]) ifTrue: [
	Error signal: 'Unexpected end of file while reading GGUF magic number' ].

"GGUF magic bytes: 0x47 0x47 0x55 0x46 == 'GGUF'"
(magic = #[71 71 85 70]) ifFalse: [
	Error signal: 'Not a GGUF file (bad magic)' ].

version := self readUInt32.
tensorCount := self readUInt64.
metadataCount := self readUInt64
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> parseMetadata [

"Parse all metadata key-value pairs"
| count |
metadata := Dictionary new.
count := self readArrayLength: metadataCount.

count timesRepeat: [
	| key valueType value |
	[
		key := self readKeyString.
		valueType := self readUInt32.
		value := self readValue: valueType.
		metadata at: key put: value
	] on: Error do: [ :ex |
		key ifNotNil: [
			Error signal: 'Error parsing metadata key ''', key, ''': ', ex messageText
		] ifNil: [
			Error signal: 'Error parsing metadata: ', ex messageText
		]
	]
]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readArray [

"Read a GGUF array: element type (uint32), element count (uint64), then elements"
| arrayType arrayLength len values |
arrayType := self readUInt32.
arrayLength := self readUInt64.
len := self readArrayLength: arrayLength.

values := Array new: len.
1 to: len do: [ :i |
	values at: i put: (self readValue: arrayType) ].

^ values
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
	| bytes value byte |

	bytes := ByteArray new: 4.
	1 to: 4 do: [ :i |
		byte := stream next.
		byte ifNil: [ Error signal: 'Unexpected end of GGUF file while reading Float32' ].
		bytes at: i put: byte ].


	"Convert little-endian bytes to integer"
	value := (bytes at: 4) << 24 + ((bytes at: 3) << 16) + ((bytes at: 2) << 8) + (bytes at: 1).

	"Convert IEEE 754 binary32 to float"
	^ Float fromIEEE32Bit: value
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readFloat64 [

	"Read a 64-bit float (little-endian)"
	| bytes value byte |

	bytes := ByteArray new: 8.
	1 to: 8 do: [ :i |
		byte := stream next.
		byte ifNil: [ Error signal: 'Unexpected end of GGUF file while reading Float64' ].
		bytes at: i put: byte ].

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

	| byte |

	byte := stream next.

byte ifNil: [ Error signal: 'Unexpected end of GGUF file while reading Int8' ].

	^ byte > 16r7F
		ifTrue: [ byte - 16r100 ]
		ifFalse: [ byte ]
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readKeyString [
"GGUF metadata keys are gguf_string_t (uint64 length) and must be ASCII per spec."
| key |
key := self readString.

(key size > 65535) ifTrue: [
	Error signal: 'Invalid GGUF key length (over 65535): ', key size asString ].

(key allSatisfy: [ :ch | ch asciiValue between: 0 and: 127 ]) ifFalse: [
	Error signal: 'Invalid GGUF key (non-ASCII): ', key ].

^ key
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readString [

"Read a GGUF string (gguf_string_t): uint64 byte length + UTF-8 bytes"
| length bytes len |
length := self readUInt64.

length > self remainingBytes ifTrue: [
	Error signal: 'Invalid GGUF string length: ', length asString ].

len := self readArrayLength: length. "ensures it fits in SmallInteger"

bytes := stream next: len.
bytes ifNil: [
	Error signal: 'Unexpected end of GGUF file while reading string' ].
(bytes size < len) ifTrue: [
	Error signal: 'Incomplete string: expected ', len asString, ' bytes, got ', bytes size asString ].

^ self decodeUtf8Bytes: bytes
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt16 [
	"Read an unsigned 16-bit integer (little-endian)"

	| b1 b2 |
	b1 := stream next.
	b1 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt16' ].
	b2 := stream next.
	b2 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt16' ].
	^ b2 << 8 + b1
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt32 [
	"Read an unsigned 32-bit integer (little-endian)"

	| b1 b2 b3 b4 |
	b1 := stream next.
	b1 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt32' ].
	b2 := stream next.
	b2 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt32' ].
	b3 := stream next.
	b3 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt32' ].
	b4 := stream next.
	b4 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt32' ].
	^ b4 << 24 + (b3 << 16) + (b2 << 8) + b1
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt64 [
	"Read an unsigned 64-bit integer (little-endian)"

	| b1 b2 b3 b4 b5 b6 b7 b8 |
	b1 := stream next.
	b1 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b2 := stream next.
	b2 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b3 := stream next.
	
	b3 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b4 := stream next.
	b4 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b5 := stream next.
	b5 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b6 := stream next.
	b6 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b7 := stream next.
	b7 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].
	b8 := stream next.
	b8 ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt64' ].

	^ b8 asInteger << 56 + (b7 asInteger << 48) + (b6 asInteger << 40) + (b5 asInteger << 32) + (b4 asInteger << 24)
	  + (b3 asInteger << 16) + (b2 asInteger << 8) + b1 asInteger
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readUInt8 [
	"Read an unsigned 8-bit integer"

	| byte |

	byte := stream next.
	byte ifNil: [ Error signal: 'Unexpected end of GGUF file while reading UInt8' ].

	^ byte
]

{ #category : 'as yet unclassified' }
AIGGUFParser >> readValue: valueType [

"GGUF metadata value types (spec):
0 UINT8, 1 INT8, 2 UINT16, 3 INT16, 4 UINT32, 5 INT32,
6 FLOAT32, 7 BOOL, 8 STRING, 9 ARRAY, 10 UINT64, 11 INT64, 12 FLOAT64"
valueType = 0 ifTrue: [ ^ self readUInt8 ].
valueType = 1 ifTrue: [ ^ self readInt8 ].
valueType = 2 ifTrue: [ ^ self readUInt16 ].
valueType = 3 ifTrue: [ ^ self readInt16 ].
valueType = 4 ifTrue: [ ^ self readUInt32 ].
valueType = 5 ifTrue: [ ^ self readInt32 ].
valueType = 6 ifTrue: [ ^ self readFloat32 ].
valueType = 7 ifTrue: [ ^ self readUInt8 ~= 0 ]. "BOOL"
valueType = 8 ifTrue: [ ^ self readString ].
valueType = 9 ifTrue: [ ^ self readArray ].
valueType = 10 ifTrue: [ ^ self readUInt64 ].
valueType = 11 ifTrue: [ ^ self readInt64 ].
valueType = 12 ifTrue: [ ^ self readFloat64 ].

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
