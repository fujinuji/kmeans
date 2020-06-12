import os

from sklearn import neural_network
from sklearn.metrics import accuracy_score

from Utils import loadData, prepareData, featureComputation, computeOutputs, readHybridWords

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
trainOutputs = computeOutputs(trainOutputs, labels)
testOutputs = computeOutputs(testOutputs, labels)

classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10, ), activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.1)
classifier.fit(trainFeatures, trainOutputs)

computedOutputs = classifier.predict(testFeatures)
print("ANN accuracy: ", accuracy_score(testOutputs, computedOutputs))