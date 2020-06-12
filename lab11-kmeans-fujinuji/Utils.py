import csv
import os

import numpy as np


def loadData():
    crtDir = os.getcwd()
    fileName = os.path.join(crtDir, 'data', 'data.csv')

    data = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    inputs = [data[i][0] for i in range(len(data))]
    outputs = [data[i][1] for i in range(len(data))]
    labelNames = list(set(outputs))

    return inputs, outputs, labelNames


def prepareData(inputs, outputs):
    np.random.seed(5)
    noSamples = len(inputs)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def featureComputation(model, data, knownWords):
    features = []
    phrases = [phrase.split() for phrase in data]
    for phrase in phrases:
        vectors = [1 + model[word] if word in knownWords else model[word] for word in phrase if (len(word) > 2) and (word in model.vocab.keys())]
        if len(vectors) == 0:
            result = [0.0] * model.vector_size
        else:
            result = np.sum(vectors, axis=0) / len(vectors)
        features.append(result)
    return features

def computeOutputs(outputs, labels):
    return [labels.index(output) for output in outputs]

def readHybridWords():
    words = []
    positiveWords = open("data/positiveWords")
    positiveWordsLines = positiveWords.readlines()

    for line in positiveWordsLines:
        words.append(line.strip())

    negativeWords = open("data/positiveWords")
    negativeWordsLines = negativeWords.readlines()

    for line in negativeWordsLines:
        words.append(line.strip())

    return words
