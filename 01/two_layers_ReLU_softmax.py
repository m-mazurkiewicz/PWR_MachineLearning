from neural_network import NeuralNetwork, sigmoid, softmax, ReLU
from mnist_loader import load_data_arrays
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    activation_functions = [ReLU, softmax]
    layers_size_vector = [784, 200, 10]
    cost_function = 'euclidean_distance'
    dropout_probabilities = None
    learning_rate = .003
    number_of_iterations = 10000
    regularization_lambda = 0
    training_data_X, training_data_Y, validation_data_X, validation_data_Y, test_data_X, test_data_Y = load_data_arrays()
    network = NeuralNetwork(layers_size_vector, activation_functions, cost_function, dropout_probabilities)
    cost = network.fit(training_data_X, training_data_Y, learning_rate, regularization_lambda, 1, number_of_iterations)
    # cost = network.fit(test_data_X, test_data_Y, learning_rate, regularization_lambda, 1, number_of_iterations)
    plt.plot(cost, 'o')
    plt.show()
