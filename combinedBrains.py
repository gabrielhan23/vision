import sys
import numpy as np
import tensorflow.keras.datasets.mnist as mnist
# import matplotlib.pyplot as plt
np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), 784)

# connections from input layer to first  layer = 100352

# activations

# dot product m1 (rows) x n1 (cols) by m2 x n2 --> n1 and m2 have to be the same


def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1 / (1 + np.exp(-x))


def snairsoftmax(array):
    vectorizedEx = np.vectorize(lambda x: np.exp(x))
    numerator = vectorizedEx(array)
    denominator = np.sum(numerator)
    return numerator/denominator


vectorizeSigmoid = np.vectorize(sigmoid)


def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))


def xadf(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


def softmax(array, derivative=False):
    # Numerically stable with large exponentials
    n = np.e**(array)
    d = sum(n)
    if derivative:
        return n
    return n / d


INPUT = 784
LAYER_1 = 128
LAYER_2 = 64
OUTPUT = 10
LEARNING_RATE = 0.01

# costtracker = []
numCorrect = 0

# a0 = x_train.reshape()  # input shape(784,)

w1 = np.random.randn(LAYER_1, INPUT, )  # shape(128,784)
b1 = np.random.randn(LAYER_1)  # shape(128)
w2 = np.random.randn(LAYER_2, LAYER_1)  # shape(64, 128)
b2 = np.random.randn(LAYER_2)  # shape(64)
w3 = np.random.randn(OUTPUT, LAYER_2)  # shape (10, 64)
b3 = np.random.randn(OUTPUT)  # shape (10)


for index, a0 in enumerate(x_train):
    # a0 = a0.reshape(784)
    a0 = np.divide(a0, 255)

    z1 = np.add(np.dot(w1, a0), b1)  # shape(128,784) ----- dotified
    a1 = vectorizeSigmoid(z1)  # shape(128,784)
    z2 = np.add(np.dot(w2, a1), b2)  # shape(128,784) ----- dotified
    a2 = vectorizeSigmoid(z2)  # shape(128,784)

    z3 = np.add(np.dot(w3, a2), b3)
    gabprediction = softmax(z3)
    prediction = snairsoftmax(z3)
    valueOfPrediction = np.argmax(prediction)
    # print(prediction, )  # prints all 1

    actual = y_train[index]
    print("prediction {0} actual {1}".format(
        valueOfPrediction, actual), end="\n")
    if actual == valueOfPrediction:
        numCorrect += 1

    # cost = (prediction.index(max(prediction)) - actual) ** 2

    # # costtracker.append(cost)

print()
print(numCorrect/60000)
# costtracker = np.array(costtracker)
# x = [range(0,len(costtracker))]
# y = costtracker
# plt.plot(x, y, color="red")
# plt.show()
