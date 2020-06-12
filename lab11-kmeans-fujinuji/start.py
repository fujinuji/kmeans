import os

import gensim
from Utils import loadData, prepareData, featureComputation, readHybridWords
from kMean import K_Means

inputs, outputs, labels = loadData()

trainInputs, trainOutputs, testInputs, testOutputs = prepareData(inputs, outputs)

# Load Google's pre-trained Word2Vec
crtDir = os.getcwd()
modelPath = os.path.join(crtDir, 'model', 'GoogleNews-vectors-negative300.bin')
print("Word2vec loaded")
knownWords = readHybridWords()
word2vecModel300 = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary=True)
trainFeatures = featureComputation(word2vecModel300, trainInputs, knownWords)
testFeatures = featureComputation(word2vecModel300, testInputs, knownWords)

print("Features extracted")

unsupervisedClassifier = K_Means()
unsupervisedClassifier.fit(trainFeatures)
computedTestIndexes = [unsupervisedClassifier.predict(feature) for feature in testFeatures]
computedTestOutputs = [labels[value] for value in computedTestIndexes]

from sklearn.metrics import accuracy_score
print("acc with code: ", accuracy_score(testOutputs, computedTestOutputs))