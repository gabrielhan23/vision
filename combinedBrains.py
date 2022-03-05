import sys
import numpy as np
import tensorflow.keras.datasets.mnist as mnist
# import matplotlib.pyplot as plt
np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), 784)
x_test = x_test.reshape(len(x_test), 784)

# connections from input layer to first  layer = 100352

# activations

# dot product m1 (rows) x n1 (cols) by m2 x n2 --> n1 and m2 have to be the same


def sigmoid(array, derivative=False):
    if derivative:
        return [(np.exp(-x))/((np.exp(-x)+1)**2) for x in array]
    return [1/(1 + np.exp(-x)) for x in array]


def softmax(array, derivative=False):
    # Numerically stable with large exponentials
    n = np.e**(array)
    d = sum(n)
    if derivative:
        return n*(d-n) / d**2
    return n / d

def relu(array, derivative=False):
    if derivative:
        s = []
        for x in array:
            if x < 0: s.append(0)
            else: s.append(1)
        return s
    return [max(0,x) for x in array]

INPUT = 784
LAYER_1 = 128
LAYER_2 = 64
OUTPUT = 10
LEARNING_RATE = 0.01
EPOCHES = 12

# costtracker = []
numCorrect = 0

# a0 = x_train.reshape()  # input shape(784,)

w1 = np.zeros((LAYER_1, INPUT))  # shape(128,784)
b1 = np.zeros(LAYER_1)  # shape(128)
w2 = np.zeros((LAYER_2, LAYER_1))  # shape(64, 128)
b2 = np.zeros(LAYER_2)  # shape(64)
w3 = np.zeros((OUTPUT, LAYER_2))  # shape (10, 64)
b3 = np.zeros(OUTPUT)  # shape (10)

tw1 = np.zeros((LAYER_1, INPUT))  # shape(128,784)
tb1 = np.zeros(LAYER_1)  # shape(128)
tw2 = np.zeros((LAYER_2, LAYER_1))  # shape(64, 128)
tb2 = np.zeros(LAYER_2)  # shape(64)
tw3 = np.zeros((OUTPUT, LAYER_2))  # shape (10, 64)
tb3 = np.zeros(OUTPUT)  # shape (10)


for epoch in range(EPOCHES):
    for index, a0 in enumerate(x_train):
        # normalize
        a0 = np.divide(a0, 255)

        z1 = np.add(np.dot(w1, a0), b1)  # shape(128,784) ----- dotified
        a1 = sigmoid(z1)  # shape(128,784)
        z2 = np.add(np.dot(w2, a1), b2)  # shape(128,784) ----- dotified
        a2 = sigmoid(z2)  # shape(128,784)

        z3 = np.add(np.dot(w3, a2), b3)
        prediction = softmax(z3)
        # prediction = snairsoftmax(z3)
        valueOfPrediction = np.argmax(prediction)
        # print(prediction, )  # prints all 1

        actualArray = np.zeros(OUTPUT)
        actual = y_train[index]
        actualArray[actual] = 1

        # print("prediction {0} actual {1}".format(valueOfPrediction, actual), end="\n")
        if actual == valueOfPrediction: numCorrect += 1

        # cost = (prediction.index(max(prediction)) - actual) ** 2

        # # costtracker.append(cost)
        # prediction - actual = shape(10)
        # softmax z3 = shape(10)

        e3 = softmax(z3, derivative=True)*(prediction-actualArray)*2 
        dcw3 = np.outer(e3,a2)
        dca3 = np.dot(w3.T,e3)
        # dca3 = np.dot(e3,w3)
        tw3 -= dcw3
        tb3 -= e3

        e2 = sigmoid(z2, derivative=True)*dca3
        dcw2 = np.outer(e2,a1)
        dca2 = np.dot(w2.T,e2)
        # dca2 = np.dot(e2,w2)
        tw2 -= dcw2
        tb2 -= e2

        e1 = sigmoid(z1, derivative=True)*dca2 
        dcw1 = np.outer(e1,a0)
        # dca1 = np.dot(w1.T,e1)
        # dca1 = np.dot(e1,w1)
        tw1 -= dcw1
        tb1 -= e1
        
        # print(sum((prediction-actualArray)**2))

    print(numCorrect/60000)
    
    w3 += tw3/60000
    w2 += tw2/60000
    w1 += tw1/60000
    b3 += tb3/60000
    b2 += tb2/60000
    b1 += tb1/60000

    tw1 = np.zeros((LAYER_1, INPUT))  # shape(128,784)
    tb1 = np.zeros(LAYER_1)  # shape(128)
    tw2 = np.zeros((LAYER_2, LAYER_1))  # shape(64, 128)
    tb2 = np.zeros(LAYER_2)  # shape(64)
    tw3 = np.zeros((OUTPUT, LAYER_2))  # shape (10, 64)
    tb3 = np.zeros(OUTPUT)  # shape (10)
    numCorrect = 0


    # costtracker = np.array(costtracker)
    # x = [range(0,len(costtracker))]
    # y = costtracker
    # plt.plot(x, y, color="red")
    # plt.show()
