Class {
	#name : 'AILlamaLibraryTest',
	#superclass : 'TestCase',
	#category : 'AI-PharoInfer-Tests-Tests',
	#package : 'AI-PharoInfer-Tests',
	#tag : 'Tests'
}

{ #category : 'running' }
AILlamaLibraryTest >> tearDown [

	AILlamaLibrary reset.
	super tearDown
]

{ #category : 'tests' }
AILlamaLibraryTest >> testDefaultLibraryNames [

	self
		assert: (AILlamaLibrary
				 libraryPathFor: 'libai_llama.dylib'
				 inDirectories: #())
		equals: 'libai_llama.dylib'.
	self
		assert: (AILlamaLibrary
				 libraryPathFor: 'libai_llama.so'
				 inDirectories: #())
		equals: 'libai_llama.so'.
	self
		assert: (AILlamaLibrary
				 libraryPathFor: 'ai_llama.dll'
				 inDirectories: #())
		equals: 'ai_llama.dll'
]

{ #category : 'tests' }
AILlamaLibraryTest >> testExplicitLibraryPathOverridesPlatformNames [

	| library path |
	path := '/opt/pharo-infer/lib/custom-ai-llama'.
	AILlamaLibrary libraryPath: path.
	library := AILlamaLibrary new.

	self assert: library macLibraryName equals: path.
	self assert: library unixLibraryName equals: path.
	self assert: library unix64LibraryName equals: path.
	self assert: library win32LibraryName equals: path
]

{ #category : 'tests' }
AILlamaLibraryTest >> testLibraryPathCanBeResolvedFromCandidateDirectories [

	| root libDir libraryFile resolved |
	root := FileLocator temp / ('pharo-infer-library-test-' , UUID new asString).
	libDir := root / 'pharo-infer-native' / 'lib'.
	libDir ensureCreateDirectory.
	libraryFile := libDir / 'ai_llama.dll'.
	libraryFile ensureCreateFile.

	[
		resolved := AILlamaLibrary
			            libraryPathFor: 'ai_llama.dll'
			            inDirectories: { libDir }.
		self assert: resolved equals: libraryFile fullName ]
		ensure: [ root ensureDeleteAll ]
]
