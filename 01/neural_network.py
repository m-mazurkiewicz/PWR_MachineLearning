import numpy as np

class NeuralNetwork:

    def __init__(self, number_of_layers, layers_size_vector, activation_function):
        if number_of_layers != len(layers_size_vector):
            raise Exception('Inconsistent input parameters!')
        self.activation_function = activation_function
        self.number_of_layers = number_of_layers
        self.initialise_parameters(layers_size_vector)

    def initialise_parameters(self, layers_size_vector):
        self.weights = dict()
        self.bias = dict()
        for i in range(self.number_of_layers):
            self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])*2-1
            self.bias[i] = np.random.rand(layers_size_vector[i],1)*2-1
