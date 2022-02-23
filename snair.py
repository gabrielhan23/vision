
import numpy as np
import tensorflow.keras.datasets.mnist as mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))


vectorizeSigmoid = np.vectorize(sigmoid)
vecotrizeSigmoidPrime = np.vectorize(sigmoidPrime)


def computeWeightedSum(weight, activation, bias):
    return np.add(np.dot(weight, activation), bias)


def feedForward(activation, weight, biases):
    return computeWeightedSum(activation=activation, weight=weight, bias=biases)


def calculateCost(prediction, actual):
    print("prediction len {0}".format(len(prediction)))
    print("actual len {0}".format(len(actual)))
    square = np.vectorize(lambda x: x**2)
    return square(np.subtract(prediction, actual))


def costDerivativeWithRespectToBiases(z, prediction, actual):
    return 2 * (calculateCost(prediction=prediction, actual=actual)) * vecotrizeSigmoidPrime(z)


def costDerivativeWithRespectToWeight(sharedDerivative, prevLayerActivation):
    return sharedDerivative * prevLayerActivation


LAYER_1 = 128
LAYER_2 = 64
OUTPUT = 10
LEARNING_RATE = 0.01

layer1Weights = np.random.randn(LAYER_1, 784)
layer2Weights = np.random.randn(LAYER_2, LAYER_1)
outputWeights = np.random.randn(OUTPUT, LAYER_2)

layer1Biases = np.random.randn(LAYER_1)
layer2Biases = np.random.randn(LAYER_2)
outputBiases = np.random.randn(OUTPUT)

EPOCH = 0

for index, value in enumerate(x_train):
    answer = y_train[index]
    input = value.reshape(784)
    # forward prop
    z1 = feedForward(
        activation=input, weight=layer1Weights, biases=layer1Biases)
    layer1Activation = vectorizeSigmoid(z1)
    z2 = feedForward(
        activation=z1, weight=layer2Weights, biases=layer2Biases)
    layer2Activation = vectorizeSigmoid(z2)

    # prediction
    outputZ = feedForward(activation=z2,
                          weight=outputWeights, biases=outputBiases)
    outputLayer = vectorizeSigmoid(outputZ)

    actual = np.zeros(10)
    actual[answer-1] = 1
    # back prop
    outputLayerBiasesDerivative = costDerivativeWithRespectToBiases(
        outputZ, outputLayer, actual)

    outputLayerWeightDerivative = costDerivativeWithRespectToWeight(
        outputLayerBiasesDerivative, outputLayer)

    layer2BiasesDerivative = costDerivativeWithRespectToBiases(
        z2,  outputLayer, actual)
    layer2WeightsDerivative = costDerivativeWithRespectToWeight(
        layer2BiasesDerivative, outputLayer)

    layer1BiasesDerivative = costDerivativeWithRespectToBiases(
        z1, outputLayer, actual)
    layer1WeightsDerivative = costDerivativeWithRespectToWeight(
        layer1BiasesDerivative, outputLayer)

    layer1Biases = np.divide(np.subtract(
        layer1Biases, layer1BiasesDerivative), LEARNING_RATE)
    layer1Weights = np.divide(np.subtract(
        layer1Weights, layer1WeightsDerivative), LEARNING_RATE)
    layer2Biases = np.divide(np.subtract(
        layer2Biases, layer2BiasesDerivative), LEARNING_RATE)
    layer2Weights = np.divide(np.subtract(
        layer2Weights, layer2WeightsDerivative), LEARNING_RATE)
    outputBiases = np.divide(np.subtract(
        outputBiases, layer1BiasesDerivative), LEARNING_RATE)
    outputLayer = np.divide(np.subtract(
        outputLayer, layer1WeightsDerivative), LEARNING_RATE)

    print("Epoch {0}".format(EPOCH))
