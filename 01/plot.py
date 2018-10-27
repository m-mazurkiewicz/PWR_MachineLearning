#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from neural_network import NeuralNetwork, sigmoid, ReLU


def sigma(x):
    return 1 / (1 + np.exp(-x))


def getFakeNeuralNetOutput(x, y):
    firstLayerNeuron1 = sigma(x + 0.01 * y)
    firstLayerNeuron2 = sigma(0.01 * x + y)
    outputLayer1 = sigma(firstLayerNeuron1 - firstLayerNeuron2 + 0.3)
    outputLayer2 = sigma(-  firstLayerNeuron1 + firstLayerNeuron2 - 0.3)
    return [outputLayer1, outputLayer2]


def getDecisionOfFakeNeuralNet(x, y):
    output = getFakeNeuralNetOutput(x, y)
    return 1 if output[1] > output[0] else 0


def getDecisionOfFakeNeuralNet_our(x, y, output_function):
    output = output_function(np.array([[x, y]]).T).flatten().tolist()
    # print(output)
    return 1 if output[1] > output[0] else 0


def getSamples(N):
    samples = []
    for i in range(N):
        x = np.random.normal()
        y = np.random.normal()
        which = 1 if x > 0 and y < 0 else 0
        samples.append((x, y, which))
    return samples


def getSamples_array(N):
    X = np.random.normal(size=(2, N))
    Y = np.zeros((2, N),dtype='int')
    for i in range(N):
        if (X[0, i] > 0) and (X[1, i] < 0):
        # if X[1, i] > 0:
            Y[1, i] = 1
        else:
            Y[0, i] = 1
    return X,Y


def plotDecisionDomain(listOfX, listOfY, decisionFunction):
    arrayOfX, arrayOfY = np.meshgrid(listOfX, listOfY)
    # print([[decisionFunction(x, y) for y in listOfY] for x in listOfX])
    plt.contourf(arrayOfX, arrayOfY, [[decisionFunction(x, y) for y in listOfY] for x in listOfX])


def plotDecisionDomain_our(listOfX, listOfY, decisionFunction, output_function):
    arrayOfX, arrayOfY = np.meshgrid(listOfX, listOfY)
    # print([[decisionFunction(np.array([[x, y]]).T) for y in listOfY] for x in listOfX])
    plt.contourf(arrayOfX, arrayOfY, [[decisionFunction(x, y, output_function) for y in listOfY] for x in listOfX])


def plotSamples(samples):
    markers = ['o', 'x']
    colors = ['red', 'gray']
    for sample in samples:
        plt.scatter(sample[0], sample[1],
                    marker=markers[sample[2]], color=colors[sample[2]],
                    alpha=0.5)

def plotSamples_array(X,Y):
    markers = ['o', 'x']
    colors = ['red', 'gray']
    for sample in range(X.shape[1]):
        plt.scatter(X[0,sample], X[1,sample],
                    marker=markers[Y[1,sample]], color=colors[Y[1,sample]],
                    alpha=0.5)

def getGrid(view):
    return [view[0] + (view[1] - view[0]) * i / (view[2] - 1) for i in range(view[2])]


numberOfSamples = 3000

viewX = [-4, 4, 101]
viewY = [-4, 4, 101]

X,Y = getSamples_array(numberOfSamples)
NN = NeuralNetwork(4, [2, 100,100,100, 2], sigmoid)
costs = NN.fit(X, Y, 0.09, 3, 0.99999, 10000)
plt.plot(costs,'o-')
plt.show()
# NN.set_weights([(np.array([[1, 0.01],[0.01, 1]]), np.array([[0],[0]])),(np.array([[1, -1],[-1, 1]]),np.array([[0.3],[-0.3]]))])
plotDecisionDomain_our(getGrid(viewX), getGrid(viewY), getDecisionOfFakeNeuralNet_our, NN.whole_output)
plotSamples_array(X,Y)
plt.show()
# samples = getSamples(numberOfSamples)
# plotDecisionDomain(getGrid(viewX), getGrid(viewY), getDecisionOfFakeNeuralNet)
# plotSamples(samples)
# plt.show()
