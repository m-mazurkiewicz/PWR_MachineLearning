import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import autograd.numpy as np_autograd
from autograd import elementwise_grad as egrad


class NeuralNetwork:

    min_max_scaler = preprocessing.MinMaxScaler()

    def __init__(self, layers_size_vector, activation_function, cost_function = 'cross-entropy'):
        self.number_of_layers = len(layers_size_vector)
        if type(activation_function) is list:
            if self.number_of_layers != len(activation_function) + 1:
                raise Exception("layer_size_vector & activation_function dimension mismatch")
            self.activation_function = activation_function
        else:
            self.activation_function = [activation_function] * self.number_of_layers
        self.initialise_parameters(layers_size_vector)
        self.layers_size_vector = layers_size_vector
        self.cache = []
        self.fitted = False
        self.cost_function = cost_function

    def initialise_parameters(self, layers_size_vector):
        self.weights = dict()
        self.bias = dict()
        for i in range(1, self.number_of_layers):
            # self.weights[i] = np.random.randn(layers_size_vector[i],layers_size_vector[i-1]) * np.sqrt(2/layers_size_vector[i-1])  #He initialisation
            self.weights[i] = np.random.randn(layers_size_vector[i],layers_size_vector[i-1]) / np.sqrt(layers_size_vector[i-1])
            self.bias[i] = np.zeros((layers_size_vector[i],1))

    def whole_output(self, A):
        self.cache_A = dict()
        self.cache_Z = dict()
        self.cache_A[0] = A
        for i in range(1,self.number_of_layers):
            Z = self.linear_forward(A, i)
            A = self.activation_function[i-1](Z)
            self.cache_A[i] = A
            self.cache_Z[i] = Z
        return A

    def linear_forward(self, previous_A, layer_no):
        return np.dot(self.weights[layer_no], previous_A)+self.bias[layer_no]

    def predict(self,input_matrix):
        output = self.whole_output(input_matrix)
        if output.shape[0] == 1:
            o = output > 0.5
            o = o[:,np.newaxis]
            return o
        else:
            return np.argmax(output, axis=0)

    def cost_function_evaluation(self, X, Y, _lambda = 0):
        output = self.whole_output(X)
        # if self.cost_function == 'cross-entropy':
        #     return np.nansum(-np.multiply(Y, np.log(output)) - np.multiply((1 - Y), np.log(1 - output))) / Y.shape[1]
        return np.nansum(-np.multiply(Y, np.log(output)) - np.multiply((1 - Y), np.log(1 - output))) / Y.shape[1]
        # elif self.cost_function == 'euclidean_distance':
        #     return 1/2 * np.sum(np.power(Y-output,2)) / Y.shape[1]

    def output_layer_cost_derivative(self, output_matrix, Y):
        if self.cost_function == 'cross-entropy':
            return - (np.divide(Y, output_matrix) - np.divide(1 - Y, 1 - output_matrix))
        elif self.cost_function == 'euclidean_distance':
            # print((output_matrix - Y).shape)
            return output_matrix - Y
        else:
            raise Exception("Wrong cost function name")

    def back_propagation(self, X, Y, regularisation_lambda = 0):
        self.cost_derivatives = dict()
        self.weight_derivatives = dict()
        self.bias_derivatives = dict()
        dZ = self.output_layer_cost_derivative(self.whole_output(X), Y)
        for i in reversed(range(1,self.number_of_layers)):
            self.weight_derivatives[i] = (np.dot(dZ, self.cache_A[i-1].T) + regularisation_lambda * self.weights[i]) / X.shape[1]
            self.bias_derivatives[i] = np.sum(dZ, axis=1, keepdims=True) / X.shape[1]
            self.cost_derivatives[i - 1] = np.dot(self.weights[i].T, dZ)
            if i>1:
                # dZ = self.cost_derivatives[i-1] * self.activation_function[i-1](self.cache_A[i-1], grad=True)
                dZ = self.cost_derivatives[i-1] * ReLU(self.cache_A[i-1], grad=True)

    def update_weights(self, learning_rate):
        for i in range(1,self.number_of_layers):
            self.weights[i] -= learning_rate * self.weight_derivatives[i]
            self.bias[i] -= learning_rate * self.bias_derivatives[i]

    def fit(self, X, Y, learning_rate, regularisation_lambda, epsilon,max_iteration_number = 10000, min_iteration_number = 4, min_max_normalization = True):
        costs = []
        if not self.fitted:
            if min_max_normalization:
                X = self.min_max_scaler.fit_transform(X)-0.5
            previous_cost_function =  float('inf')
            counter = 0
            # while ((self.cost_function_evaluation(X, Y) / previous_cost_function <= epsilon) and (counter<max_iteration_number)) or (counter<min_iteration_number):
            while counter<max_iteration_number:
                previous_cost_function = self.cost_function_evaluation(X,Y)
                self.whole_output(X)
                self.back_propagation(X, Y, regularisation_lambda)
                self.update_weights(learning_rate)
                counter +=1
                costs.append(self.cost_function_evaluation(X,Y))
            self.fitted = True
            return costs
        else:
            raise Exception("Neural network already fitted!")

    def set_weights(self, list_of_parameters):
        for layer_no, parameters in enumerate(list_of_parameters):
            self.weights[layer_no] = parameters[0]
            self.bias[layer_no] = parameters[1]


def ReLU(x, grad = False):
    if grad == True:
        return np.int64(x > 0)
    # return x * (x > 0)
    return np.maximum(0, x)


def sigmoid(x, grad = False):
    s = 1 / (1 + np.exp(-x))
    if grad == True:
        return s * (1 - s)
    return s


def softmax(x, grad = False):
    def softmax_eval(x):
        e_x = np_autograd.exp(x - np_autograd.max(x))
        return e_x / e_x.sum(axis=0)
    softmax_eval_grad = egrad(softmax_eval)
    if grad:
        return softmax_eval_grad(x)
    else:
        return softmax_eval(x)

def getSamples_array(N):
    X = np.random.normal(size=(2, N))
    Y = np.zeros((N,1))
    for i in range(N):
        if (X[0, i] > 0) and (X[1, i] < 0):
            Y[i] = 1
    return X,Y

if __name__ == '__main__':
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
    X,Y = getSamples_array(300)
    NN = NeuralNetwork([2,20,3,1],[sigmoid, softmax],'cross-entropy')
    costs = NN.fit(X, Y.T, 0.3, 0, 0.9995, 30000, min_max_normalization=False)
    plt.plot(costs,'o-')
    plt.show()
    # print(NN.whole_output(X))
    # NN.set_weights([(np.array([[1, 0.01],[0.01, 1]]), np.array([[0],[0]])),(np.array([[1, -1],[-1, 1]]),np.array([[0.3],[-0.3]]))])
    # NN.predict(X)
    # print(NN.whole_output(X))
    # print(NN.predict(X).any())
    # print(NN.predict(X).all())
    # print(Y[1,:].any())