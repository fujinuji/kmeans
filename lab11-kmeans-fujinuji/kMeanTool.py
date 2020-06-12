import os

from sklearn.cluster import KMeans

from Utils import loadData, prepareData, featureComputation, readHybridWords

inputs, outputs, labels = loadData()

trainInputs, trainOutputs, testInputs, testOutputs = prepareData(inputs, outputs)

import gensim

# Load Google's pre-trained Word2Vec
crtDir = os.getcwd()
modelPath = os.path.join(crtDir, 'model', 'GoogleNews-vectors-negative300.bin')
print("Word2vec loaded")
knownWords = readHybridWords()
word2vecModel300 = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary=True)
trainFeatures = featureComputation(word2vecModel300, trainInputs, knownWords)
testFeatures = featureComputation(word2vecModel300, testInputs, knownWords)

print("Features extracted")

unsupervisedClassifier = KMeans(n_clusters=2, random_state=0)
unsupervisedClassifier.fit(trainFeatures)
computedTestIndexes = unsupervisedClassifier.predict(testFeatures)
computedTestOutputs = [labels[value] for value in computedTestIndexes]

from sklearn.metrics import accuracy_score
print("acc with tool: ", accuracy_score(testOutputs, computedTestOutputs))
