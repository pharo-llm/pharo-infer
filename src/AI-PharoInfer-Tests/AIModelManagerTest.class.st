Class {
	#name : 'AIModelManagerTest',
	#superclass : 'TestCase',
	#instVars : [
		'manager'
	],
	#category : 'AI-PharoInfer-Tests',
	#package : 'AI-PharoInfer-Tests'
}

{ #category : 'running' }
AIModelManagerTest >> setUp [
	super setUp.
	manager := AIModelManager new
]

{ #category : 'running' }
AIModelManagerTest >> tearDown [

	manager unloadAll.
	super tearDown
]

{ #category : 'running' }
AIModelManagerTest >> testDiscoverModels [

	| testDir discovered |

	testDir := FileLocator temp / 'pharo-test-models'.
	testDir ensureCreateDirectory.

	[ (testDir / 'model1.gguf') ensureCreateFile.
	(testDir / 'model2.gguf') ensureCreateFile.

	discovered := manager discoverModelsInDirectory: testDir.

	self assert: discovered size equals: 2.
	self assert: (manager listModels includes: 'model1').
	self assert: (manager listModels includes: 'model2') ]
		ensure: [ testDir ensureDeleteAll ]
]

{ #category : 'running' }
AIModelManagerTest >> testListModels [

	| model1 model2 |

	model1 := AIModel named: 'model1'.
	model2 := AIModel named: 'model2'.

	manager registerModel: model1.
	manager registerModel: model2.

	self assert: manager listModels size equals: 2.
	self assert: (manager listModels includes: 'model1').
	self assert: (manager listModels includes: 'model2')
]

{ #category : 'running' }
AIModelManagerTest >> testLoadModel [

	| model file |

	file := FileLocator temp / 'test-load-model.gguf'.
	file ensureCreateFile.

	[ model := AIModel fromFile: file.
	model backend: AILocalBackend new.
	manager registerModel: model.

	manager loadModel: 'test-load-model'.

	self assert: model isLoaded ]
		ensure: [ file ensureDelete ]
]

{ #category : 'running' }
AIModelManagerTest >> testRegisterModel [

	| model |

	model := AIModel named: 'test-model'.
	manager registerModel: model.

	self assert: (manager modelNamed: 'test-model') equals: model
]

{ #category : 'running' }
AIModelManagerTest >> testUnloadAll [

	| model1 model2 file1 file2 |

	file1 := FileLocator temp / 'model1.gguf'.
	file2 := FileLocator temp / 'model2.gguf'.
	file1 ensureCreateFile.
	file2 ensureCreateFile.

	[ model1 := AIModel fromFile: file1.
	model1 backend: AILocalBackend new.
	model2 := AIModel fromFile: file2.
	model2 backend: AILocalBackend new.

	manager registerModel: model1.
	manager registerModel: model2.

	model1 load.
	model2 load.

	manager unloadAll.

	self deny: model1 isLoaded.
	self deny: model2 isLoaded ]
		ensure: [
			file1 ensureDelete.
			file2 ensureDelete ]
]

{ #category : 'running' }
AIModelManagerTest >> testUnregisterModel [

	| model |

	model := AIModel named: 'test-model'.
	manager registerModel: model.

	manager unregisterModel: 'test-model'.

	self assert: (manager modelNamed: 'test-model') isNil
]
