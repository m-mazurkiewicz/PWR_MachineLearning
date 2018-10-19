import numpy as np


class NeuralNetwork:

    def __init__(self, number_of_layers, layers_size_vector, activation_function):
        if number_of_layers != len(layers_size_vector)-1:
            raise ValueError('Inconsistent input parameters!')
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
        input_vector = np.vstack([1,input_vector])
        # print(input_vector)
        if len(input_vector) != self.layers_size_vector[0]:
            raise ValueError('Dimensions mismatch!')
        A = input_vector
        for i in range(self.number_of_layers):
            # print(A)
            # print(self.weights[1].shape, A.shape)
            A = self.activation_function(np.dot(self.weights[i], A) + self.bias[i])
        return A.flatten().tolist()

    def predict(self,input_vector):
        pass

    def cost_function(self, x, y, _lambda = 0):
        return sum(sum((-y)))

    def fit(self, learning_rate, epsilon):
        # self.cost_function
        pass




def ReLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    NN = NeuralNetwork(3,[10,20,20,2],sigmoid)
    print(NN.output((np.ones((9,1))*10)))
