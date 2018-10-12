import numpy as np
import math


class NeuralNetwork:

    def __init__(self, number_of_layers, layers_size_vector, activation_function):
        if number_of_layers != len(layers_size_vector)-1:
            raise Exception('Inconsistent input parameters!')
        self.activation_function = activation_function
        self.number_of_layers = number_of_layers
        self.initialise_parameters(layers_size_vector)
        self.layers_size_vector = layers_size_vector

    def initialise_parameters(self, layers_size_vector):
        self.weights = dict()
        self.bias = dict()
        for i in range(self.number_of_layers):
            self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])*2-1
            self.bias[i] = np.random.rand(layers_size_vector[i+1],1)*2-1

    def output(self,input_vector):
        input_vector = [1] + input_vector
        if len(input_vector) != self.layers_size_vector[0]:
            raise Exception('Dimensions mismatch!')
        A = input_vector
        for i in range(self.number_of_layers):
            A = self.activation_function(np.dot(self.weights[i], A) + self.bias[i])
        return A

def ReLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    NN = NeuralNetwork(3,[10,20,20,2],sigmoid)
    print(NN.output(np.random.rand(10,1)*10))
