import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt


class NeuralNetwork:

    min_max_scaler = preprocessing.MinMaxScaler()

    def __init__(self, number_of_layers, layers_size_vector, activation_function, cost_function = 'cross-entropy'):
        if number_of_layers != len(layers_size_vector)-1:
            raise ValueError('Inconsistent input parameters!')
        #self.activation_function = activation_function
        self.number_of_layers = number_of_layers
        self.initialise_parameters(layers_size_vector)
        self.layers_size_vector = layers_size_vector
        self.cache = []
        self.fitted = False
        self.cost_function = cost_function

    def initialise_parameters(self, layers_size_vector):
        self.weights = dict()
        self.bias = dict()
        for i in range(self.number_of_layers):
            self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])
            # self.weights[i] = np.random.rand(layers_size_vector[i+1],layers_size_vector[i])
            # self.weights[i] = np.zeros((layers_size_vector[i+1],layers_size_vector[i]))
            self.bias[i] = np.zeros((layers_size_vector[i+1], 1))
            #self.bias[i] = np.random.rand(layers_size_vector[i+1],1)*2-1

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
        for i in range(self.number_of_layers-1):
            Z = self.linear_forward(A, i)
            self.cache.append((A, Z))
            A = ReLU(Z)
        Z = self.linear_forward(A, self.number_of_layers-1)
        self.cache.append((A, Z))
        A = sigmoid(Z)
        return A

    def linear_forward(self, previous_A, layer_no):
        # print(np.dot(self.weights[layer_no], previous_A).shape,self.bias[layer_no].shape)
        return np.dot(self.weights[layer_no], previous_A)+self.bias[layer_no]

    # def predict(self,input_matrix):
    #     output = self.whole_output(input_matrix)
    #     # print(output)
    #     return np.argmax(output, axis=0)

    def cost_function_evaluation(self, X, Y, cost_function):
        output = self.whole_output(X)
        #if cost_function == 'cross-entropy':
        #    return np.sum(-np.multiply(Y, np.log(output)) - np.multiply((1 - Y), np.log(1 - output))) / len(Y)
        #elif cost_function == 'euclidean_distance':
        return 1/2 * np.sum(np.power(Y-output,2)) / len(Y)

    def output_layer_cost_derivative(self, output_matrix, Y, cost_function):
 #       if cost_function == 'cross-entropy':
 #           return - (np.divide(Y, output_matrix.T) - np.divide(1 - Y, 1 - output_matrix.T))
  #      elif cost_function == 'euclidean_distance':
        return -(Y - output_matrix.T)

    def back_propagation(self, X, Y, regularisation_lambda = 0): #To use regularization we should change backpropagation as well. Dont know if its taken into consideration
        self.cost_derivatives = dict()
        self.weight_derivatives = dict()
        self.bias_derivatives = dict()
        self.cost_derivatives[self.number_of_layers - 1] = self.output_layer_cost_derivative(self.whole_output(X), Y, 'cross-entropy')
        dZ = self.cost_derivatives[self.number_of_layers - 1].T
        for i in reversed(range(self.number_of_layers)):
            self.weight_derivatives[i] = (np.dot(dZ, self.cache[i][0].T)) / X.shape[1]
            # self.bias_derivatives[i] = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / X.shape[1]
            self.bias_derivatives[i] = np.sum(dZ, axis=1, keepdims=True) / X.shape[1]
            self.cost_derivatives[i - 1] = np.dot(self.weights[i].T, dZ)
            dZ = self.cost_derivatives[i-1] * ReLU(self.cache[i-1][1], grad=True)

    def update_weights(self, learning_rate):
        for i in range(self.number_of_layers):
            self.weights[i] -= learning_rate * self.weight_derivatives[i]
            # print(self.bias_derivatives[i],self.bias[i])
            self.bias[i] -= learning_rate * self.bias_derivatives[i]

    def fit(self, X, Y, learning_rate, regularisation_lambda, epsilon,max_iteration_number = 2000, min_max_normalization = False):
        regularisation_lambda = 0
        costs = []
        if not self.fitted:
            if min_max_normalization:
                X = self.min_max_scaler.fit_transform(X)-0.5
            previous_cost_function = float('inf')
            counter = 0
            #while ((self.cost_function_evaluation(X, Y) / previous_cost_function <= epsilon) and (counter<max_iteration_number)) or (counter<5):
            for i in range(max_iteration_number):
                # print(counter, previous_cost_function)
                previous_cost_function = self.cost_function_evaluation(X,Y, 'cross-entropy')
                self.back_propagation(X, Y, regularisation_lambda)
                self.update_weights(learning_rate)
                #counter +=1
                # print(counter, self.cost_function(X,Y)/previous_cost_function)
                costs.append(self.cost_function_evaluation(X,Y, 'cross-entropy'))
                if i % 1000 == 0:
                    print("Cost after iteration {}: {}".format(i, costs[i]))
            self.fitted = True
            return costs
        else:
            raise Exception("Neural network already fitted!")

    def predict(self, A):
        result = []
        A = A.T
        for i in range(len(A)):
            if A[i] >= .5:
                result.append(1)
            else:
                result.append(0)
        return result

    # def set_weights(self, list_of_parameters):
    #     for layer_no, parameters in enumerate(list_of_parameters):
    #         self.weights[layer_no] = parameters[0]
    #         self.bias[layer_no] = parameters[1]


def ReLU(x, grad = False):
    if grad == True:
        return x>0
    return x * (x > 0)


def sigmoid(x, grad = False):
    s = 1 / (1 + np.exp(-x))
    if grad == True:
        return s * (1 - s)
    return s

def getSamples_array(N):
    X = np.random.normal(size=(2, N))
    Y = np.zeros((N,1))#, dtype='int')
    for i in range(N):
        if (X[0, i] > 0) and (X[1, i] < 0):
            Y[i] = 1
    return X,Y

if __name__ == '__main__':
    X, Y = getSamples_array(100)
    NN = NeuralNetwork(3, [X.shape[0], 20, 3, 1], sigmoid, 'cross-entropy')
    #NN.initialise_parameters(NN.layers_size_vector)
    # A = NN.whole_output(X)
    # NN.back_propagation(X, Y)
    # NN.update_weights(.01)
    NN.fit(X, Y, .01, 1,1)
    result = NN.whole_output(X)
    output = NN.predict(result)
    print(output)



    # print(NN.single_output((np.ones((9, 1)) * 10)))
    # print(np.multiply(NN.whole_output(np.ones((9, 3)) * 10),np.ones((2,3))))
    # a = np.array([1, 0, 1])
    # b = np.zeros((3, 2))
    # b[np.arange(3), a] = 1
    # #print(NN.cost_function(np.random.rand(9, 3) * 10, b.T))
    # #print(NN.output_layer_cost_derivative(NN.whole_output(np.random.rand(9, 3) * 10), b.T))
    # #print()
    # #NN.whole_output(np.random.rand(9, 3) * 10)
    # #print(NN.cache)
    # #NN.back_propagation(, b.T)
    # X,Y = getSamples_array(300)
    # NN = NeuralNetwork(2,[2,2,2],sigmoid,'cross-entropy')
    # # costs = NN.fit(X, Y, 0.01, 0, 0.9995, 1000)
    # # plt.plot(costs,'o-')
    # # plt.show()
    # # NN.set_weights([(np.array([[1, 0.01],[0.01, 1]]), np.array([[0],[0]])),(np.array([[1, -1],[-1, 1]]),np.array([[0.3],[-0.3]]))])
    # NN.predict(X)
    # print(NN.whole_output(X))
    # print(NN.predict(X).any())
    # print(NN.predict(X).all())
    # print(Y.any())