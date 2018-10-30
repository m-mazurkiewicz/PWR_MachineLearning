import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import autograd.numpy as np_autograd
from autograd import jacobian, grad
from autograd import elementwise_grad as egrad


class NeuralNetwork:

    min_max_scaler = preprocessing.MinMaxScaler()

    def __init__(self, layers_size_vector, activation_function, cost_function = 'cross-entropy'):
        self.number_of_layers = len(layers_size_vector)
        if type(activation_function) is list:
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
            # self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])*2-1
            self.weights[i] = np.random.randn(layers_size_vector[i],layers_size_vector[i-1]) / np.sqrt(layers_size_vector[i-1])
            # self.weights[i] = np.zeros((layers_size_vector[i+1],layers_size_vector[i]))
            # self.bias[i] = np.random.rand(layers_size_vector[i+1],1)
            self.bias[i] = np.zeros((layers_size_vector[i],1))

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

    def whole_output(self, A):
        # number_of_training_examples = input_matrix.shape[1]
        # A = np.vstack([[1] * number_of_training_examples, input_matrix])
        # for i in range(self.number_of_layers-1):
        #     Z = self.linear_forward(A, i)
        #     self.cache.append((A, Z))
        #     A = ReLU(Z)
        # Z = self.linear_forward(A,self.number_of_layers-1)
        # self.cache.append((A,Z))
        # A = sigmoid(Z)

        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(self.weights[1], A) + self.bias[1]
        A1 = ReLU(Z1)
        Z2 = np.dot(self.weights[2], A1) + self.bias[2]
        A2 = ReLU(Z2)
        Z3 = np.dot(self.weights[3], A2) + self.bias[3]
        A3 = sigmoid(Z3)

        self.cache = (Z1, A1, self.weights[1], self.bias[1], Z2, A2, self.weights[2], self.bias[2], Z3, A3, self.weights[3], self.bias[3])
        return A3

    def linear_forward(self, previous_A, layer_no):
        # print(np.dot(self.weights[layer_no], previous_A).shape,self.bias[layer_no].shape)
        return np.dot(self.weights[layer_no], previous_A)+self.bias[layer_no]

    def predict(self,input_matrix):
        output = self.whole_output(input_matrix)
        # if input_matrix.shape[0] == 1:
        #     o = output > 0.5
        #     o = o[:,np.newaxis]
        #     return o.T
        # else:
        #     # print(output)
        #     return np.argmax(output, axis=0)
        return output > 0.5

    def cost_function_evaluation(self, X, Y, _lambda = 0):
        output = self.whole_output(X)
        if self.cost_function == 'cross-entropy':
            return np.nansum(-np.multiply(Y, np.log(output)) - np.multiply((1 - Y), np.log(1 - output))) / Y.shape[1]
        elif self.cost_function == 'euclidean_distance':
            return 1/2 * np.sum(np.power(Y-output,2)) / Y.shape[1]

    def output_layer_cost_derivative(self, output_matrix, Y):
        return -(Y - output_matrix)
        # if self.cost_function == 'cross-entropy':
        #     return - (np.divide(Y, output_matrix) - np.divide(1 - Y, 1 - output_matrix))
        # elif self.cost_function == 'euclidean_distance':
        #     return -(Y - output_matrix)

    def back_propagation(self, X, Y, regularisation_lambda = 0):
        # self.cost_derivatives = dict()
        self.weight_derivatives = dict()
        self.bias_derivatives = dict()
        # self.cost_derivatives[self.number_of_layers - 1] = self.output_layer_cost_derivative(self.whole_output(X), Y)
        # dZ = self.cost_derivatives[self.number_of_layers - 1]
        # for i in reversed(range(self.number_of_layers)):
        #     self.weight_derivatives[i] = (np.dot(dZ, self.cache[i][0].T) + regularisation_lambda * self.weights[i]) / X.shape[1]
        #     # self.bias_derivatives[i] = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / X.shape[1]
        #     self.bias_derivatives[i] = np.sum(dZ, axis=1, keepdims=True) / X.shape[1]
        #     self.cost_derivatives[i - 1] = np.dot(self.weights[i].T, dZ)
        #     dZ = self.cost_derivatives[i-1] * ReLU(self.cache[i][0])#self.activation_function[i](self.cache[i][1], grad = True)
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = self.cache

        dZ3 = A3 - Y
        dW3 = 1. / m * np.dot(dZ3, A2.T)
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1. / m * np.dot(dZ2, A1.T)
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1. / m * np.dot(dZ1, X.T)
        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

        # gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
        #              "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
        #              "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        self.weight_derivatives[1] = dW1
        self.bias_derivatives[1] = db1
        self.weight_derivatives[2] = dW2
        self.bias_derivatives[2] = db2
        self.weight_derivatives[3] = dW3
        self.bias_derivatives[3] = db3

    def update_weights(self, learning_rate):
        for i in range(1,self.number_of_layers):
            self.weights[i] -= learning_rate * self.weight_derivatives[i]
            # print(self.bias_derivatives[i],self.bias[i])
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
                # print(counter, previous_cost_function)
                previous_cost_function = self.cost_function_evaluation(X,Y)
                self.whole_output(X)
                self.back_propagation(X, Y, regularisation_lambda)
                self.update_weights(learning_rate)
                counter +=1
                # print(counter, self.cost_function(X,Y)/previous_cost_function)
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
    return np.maximum(0,x)


def sigmoid(x, grad = False):
    s = 1 / (1 + np.exp(-x))
    if grad == True:
        return s * (1 - s)
    return s


def softmax(x, grad = False):
    def softmax_eval(x):
        e_x = np_autograd.exp(x - np_autograd.max(x))
        return e_x / e_x.sum(axis=0)
        # y = np_autograd.exp(-2.0 * x)
        # return (1.0 - y) / (1.0 + y)
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