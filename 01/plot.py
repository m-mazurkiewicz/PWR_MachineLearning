#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from neural_network import NeuralNetwork, sigmoid, ReLU, softmax
from sklearn.metrics import confusion_matrix

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
    output = output_function(np.array([[y, x]]).T).flatten().tolist()
    # print(output)
    # print(np.abs(output[1]-output[0]))
    # return 1 if output[1] > output[0] else 0
    # if output[0]==1:
    #     print("Yay!")
    #     return 1
    # print(output[0])
    # print(int(output[0]))
    return int(output[0])

def getSamples(N):
    samples = []
    for i in range(N):
        x = np.random.normal()
        y = np.random.normal()
        which = 1 if x > 0 and y < 0 else 0
        samples.append((x, y, which))
    return samples


# def getSamples_array(N, balanced = True):
#     X = np.random.normal(size=(2, N))
#     Y = np.zeros((2, N), dtype='int')
#     for i in range(N):
#         if (X[0, i] > 0) and (X[1, i] < 0):
#         # if X[1, i] > 0:
#             Y[1, i] = 1
#         else:
#             Y[0, i] = 1
#     if balanced:
#         X_extra = np.repeat(X[:,Y[1,:]==1],3,axis=1)
#         Y_extra = np.zeros(X_extra.shape, dtype='int')
#         Y_extra[1,:] = 1
#         X = np.hstack((X,X_extra))
#         Y = np.hstack((Y,Y_extra))
#     return X,Y

def getSamples_array(N):
    X = np.random.normal(size=(2, N))
    Y = np.zeros((1, N), dtype='int')
    for i in range(N):
        if (X[0, i] > 0) and (X[1, i] < 0):
            Y[0,i] = 1
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
                    marker=markers[Y[0,sample]], color=colors[Y[0,sample]],
                    alpha=0.5)

def getGrid(view):
    return [view[0] + (view[1] - view[0]) * i / (view[2] - 1) for i in range(view[2])]


numberOfSamples = 100

viewX = [-4, 4, 101]
viewY = [-4, 4, 101]

# X,Y = getSamples_array(numberOfSamples, balanced=True)
X,Y = getSamples_array(numberOfSamples)
# NN = NeuralNetwork(3, [2, 10, 10, 2], sigmoid,'cross-entropy')
# NN = NeuralNetwork(2, [2, 10, 2], ReLU,'euclidean_distance')
# NN = NeuralNetwork(3, [2, 20, 20, 2], [ReLU,sigmoid, ReLU],'euclidean_distance')
# NN = NeuralNetwork(2, [2, 200, 2], sigmoid,'euclidean_distance')
# NN = NeuralNetwork(2, [2,2,2], sigmoid, 'euclidean_distance')
# NN.set_weights([(np.array([[1., 0.01],[0.01, 1.]]), np.array([[0.],[0.]])),(np.array([[1., -1.],[-1., 1.]]),np.array([[0.3],[-0.3]]))])
# costs = NN.fit(X, Y, 0.001, 1, 0.9999, 10000)
NN = NeuralNetwork([2, 20, 3, 1], sigmoid, 'cross-entropy')
costs = NN.fit(X, Y, 0.3, 0, 0.9995, 30000, min_max_normalization=False)
plt.plot(costs,'o-')
plt.show()
# NN = NeuralNetwork(2, [2, 2, 2], sigmoid)
# NN.set_weights([(np.array([[1, 0.01],[0.01, 1]]), np.array([[0],[0]])),(np.array([[1, -1],[-1, 1]]),np.array([[0.3],[-0.3]]))])
plotDecisionDomain_our(getGrid(viewX), getGrid(viewY), getDecisionOfFakeNeuralNet_our, NN.predict)
plotSamples_array(X,Y)
plt.show()
# print(Y.shape, NN.predict(X).shape)
# print(np.array(Y, dtype=bool))
# predict = (NN.predict(X)-1)*(-1)
predict = NN.predict(X)
print(len(np.where(predict & np.array(Y, dtype=bool))[0]))
print(numberOfSamples-len(np.where(predict | np.array(Y, dtype=bool))[0]))
# print(confusion_matrix(np.array(Y, dtype=bool),NN.predict(X)))

# samples = getSamples(numberOfSamples)
# plotDecisionDomain(getGrid(viewX), getGrid(viewY), getDecisionOfFakeNeuralNet)
# plotSamples(samples)
# plt.show()
