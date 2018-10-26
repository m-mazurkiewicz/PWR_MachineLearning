import numpy as np


class NeuralNetwork:

    def __init__(self, number_of_layers, layers_size_vector, activation_function):
        if number_of_layers != len(layers_size_vector)-1:
            raise ValueError('Inconsistent input parameters!')
        self.activation_function = activation_function
        self.number_of_layers = number_of_layers
        self.initialise_parameters(layers_size_vector)
        self.layers_size_vector = layers_size_vector
        self.cache = []

    def initialise_parameters(self, layers_size_vector):
        self.weights = dict()
        #self.bias = dict()
        for i in range(self.number_of_layers):
            self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])*2-1
         #   self.bias[i] = np.random.rand(layers_size_vector[i+1],1)*2-1

    # def single_output(self, input_vector):
    #     input_vector = np.vstack([1,input_vector])
    #     # print(input_vector)
    #     if len(input_vector) != self.layers_size_vector[0]:
    #         raise ValueError('Dimensions mismatch!')
    #     A = input_vector
    #     for i in range(self.number_of_layers):
    #         # print(A)
    #         # print(self.weights[1].shape, A.shape)
    #         #A = self.activation_function(np.dot(self.weights[i], A))# + self.bias[i])
    #         A,_ = self.linear_forward(A, i)
    #     return A.flatten().tolist()

    def whole_output(self, input_matrix):
        number_of_training_examples = input_matrix.shape[1]
        A = np.vstack([[1] * number_of_training_examples, input_matrix])
        for i in range(self.number_of_layers):
            Z = self.linear_forward(A, i)
            self.cache.append((A, Z))
            A = self.activation_function(Z)
        return A

    def predict(self,input_matrix):
        output, _ = self.whole_output(input_matrix)
        return np.argmax(output, axis=0)

    def cost_function(self, X, Y, _lambda = 0):
        output,_ = self.whole_output(X)
        return np.sum(-np.multiply(Y, np.log(output)) - np.multiply((1 - Y), np.log(1 - output))) / Y.shape[1]

    def output_layer_cost_derivative(self, output_matrix, Y):
        return - (np.divide(Y, output_matrix) - np.divide(1 - Y, 1 - output_matrix))

    def linear_forward(self, previous_A, layer_no):
        return np.dot(self.weights[layer_no], previous_A)

    def back_propagation(self, X, Y):
        self.cost_derivatives = dict()
        self.weight_derivatives = dict()
        output_matrix = self.whole_output(X)
        self.cost_derivatives[self.number_of_layers - 1] = self.output_layer_cost_derivative(output_matrix, Y)
        for i in reversed(range(self.number_of_layers)):
            dZ = self.cost_derivatives[i] * self.activation_function(self.cache[i][1], grad = True)
            self.weight_derivatives[i] = np.dot(dZ, self.cache[i][0].T) / X.shape[1]
            self.cost_derivatives[i - 1] = np.dot(self.weights[i].T, dZ)

    def update_weights(self, learning_rate):
        for i in range(self.number_of_layers):
            self.weights[i] -= learning_rate * self.weight_derivatives[i]

    def fit(self, X, Y, learning_rate, epsilon, max_iteration_number = 10000):
        previous_cost_function = float('inf')
        counter = 0
        while (self.cost_function(X, Y) / previous_cost_function < epsilon) and (counter<max_iteration_number):
            self.back_propagation(X, Y)
            self.update_weights(learning_rate)
            counter +=1


def ReLU(x, grad = False):
    if grad == True:
        return x>0
    return x * (x > 0)


def sigmoid(x, grad = False):
    s = 1 / (1 + np.exp(-x))
    if grad == True:
        return s * (1 - s)
    return s

if __name__ == '__main__':
    NN = NeuralNetwork(3,[10,20,30,2],sigmoid)
    # print(NN.single_output((np.ones((9, 1)) * 10)))
    # print(np.multiply(NN.whole_output(np.ones((9, 3)) * 10),np.ones((2,3))))
    a = np.array([1, 0, 1])
    b = np.zeros((3, 2))
    b[np.arange(3), a] = 1
    #print(NN.cost_function(np.random.rand(9, 3) * 10, b.T))
    #print(NN.output_layer_cost_derivative(NN.whole_output(np.random.rand(9, 3) * 10), b.T))
    #print()
    #NN.whole_output(np.random.rand(9, 3) * 10)
    #print(NN.cache)
    #NN.back_propagation(, b.T)
    NN.fit(np.random.rand(9, 3) * 10, b.T, 3, .9, 100)
