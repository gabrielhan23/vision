import tensorflow
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.datasets import mnist

#hi there
(x_train, y_train), (x_val, y_val) = mnist.load_data('/Users/gabrielhan/Coding/vision/mnist.npz')


print("god damn")

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, array, derivative=False):
        if derivative:
            return (np.exp(-array))/((np.exp(-array)+1)**2)
        return 1/(1 + np.exp(-array))

    def softmax(self, array, derivative=False):
        # Numerically stable with large exponentials
        if derivative:
            return np.e**(array)*(sum(np.e**(array))-np.e**(array)) / sum(np.e**(array))**2
        return np.e**(array) / sum(np.e**(array))

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.rand(hidden_1, input_layer)-0.5,
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train
        # print("FORWARD PASS A0: ",params['A0'])

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        print("FORWARD PASS w1: ",params['W1'])
        print("FORWARD PASS z1: ",params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
        print("FORWARD PASS A1: ",params['A1'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])
        # print("FORWARD PASS A3: ",params['A3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            ?? = ?? - ?? * ???J(x, y), 
                theta ??:            a network parameter (e.g. a weight w)
                eta ??:              the learning rate
                gradient ???J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta ??
        '''
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            # print("Output:",output)
            pred = np.argmax(output)
            # print("Pred: ", pred)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                x = x.reshape(784)
                output = self.forward_pass(x)
                print("OUTPUT: ",output)
                print("Predicted: ",np.argmax(output))
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            x_val = x_val.reshape(10000,784)
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
            
dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])

dnn.train(x_train, y_train, x_val, y_val)