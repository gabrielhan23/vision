from re import A
import numpy as np
import testCases
import math
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plot 


x, y = mnist.load_data('/Users/gabrielhan/Coding/vision/mnist.npz')



def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x, d, derivative=False):
    # Numerically stable with large exponentials
    n = np.e**(x)
    if derivative:
        return n
    return n / d

def relu(x, derivative=False):
    if derivative:
        if x < 0: return 0
        return 1
    return max(0,x)

LAYER_1 = 128
LAYER_2 = 64
LAYER_3 = 10
CHANGE = 0.1

w1 = np.random.randn(LAYER_1,len(x[0][0])*len(x[0][0][0])) # previous layer nodes by current layer
w2 = np.random.randn(LAYER_2,LAYER_1)
w3 = np.random.randn(LAYER_3,LAYER_2)

b1 = np.random.randn(LAYER_1) # previous layer nodes by current layer
b2 = np.random.randn(LAYER_2)
b3 = np.random.randn(LAYER_3)

correct = 0
batch = 0
tw1 = np.random.randn(LAYER_1,len(x[0][0])*len(x[0][0][0]))
tw2 = np.random.randn(LAYER_2,LAYER_1)
tw3 = np.random.randn(LAYER_3,LAYER_2)
tb1 = np.zeros(LAYER_1) 
tb2 = np.zeros(LAYER_2) 
tb3 = np.zeros(LAYER_3) 

for it, input in enumerate(x[0]):
    # plot.imshow(input, cmap='gray')
    # plot.show()
    answer = x[1][it]
    # forward prop

    a0 = input.reshape(len(input)*len(input[0]))
    a0 = np.divide(a0, 255.)

    
    z1 = np.dot(w1,a0) - b1
    a1 = [relu(x) for x in z1]
    print(a1)

    z2 = np.dot(w2,a1) - b2
    a2 = [relu(x) for x in z2]

    z3 = np.dot(w3,a2) - b3
    as3 = sum(np.e**z3)
    a3 = [softmax(x,as3) for x in z3]
    print(a2)
    print(a3)
    ye = np.zeros(LAYER_3)
    ye[answer] = 1
    print(a3)
    print("prediction {0} actual {1}".format(np.argmax(a3), answer))

    # back prop
    ec3 = sum(np.e**z3)
    print("AHHHHHH  ",[softmax(x,ec3,derivative=True) for x in z3])
    e3 = [softmax(x,ec3,derivative=True) for x in z3]*(a3-ye)*2
    dc3 = np.outer(e3,a2)
    da3 = np.dot(w3.T,e3)
    tw3 += dc3
    tb3 += e3

    e2 = [relu(x,derivative=True) for x in z2]*da3*2
    dc2 = np.outer(e2,a1)
    da2 = np.dot(w2.T,e2)
    tw2 += dc2
    tb2 += e2

    e1 = [relu(x,derivative=True) for x in z1]*da2*2
    dc1 = np.outer(e1,a0)
    da1 = np.dot(w1.T,e1)
    tw1 += dc1
    tb1 += e1

    if batch > 100: 
        w3 -= tw3/batch*CHANGE
        w2 -= tw2/batch*CHANGE
        w1 -= tw1/batch*CHANGE
        b3 -= tb3/batch*CHANGE
        b2 -= tb2/batch*CHANGE
        b1 -= tb1/batch*CHANGE
        
        batch = 0
    else: 
        batch += 1

    # if answer == a3.index(max(a3))+1: correct+=1
    # print(correct/(it+1)*100," percent correct         \r",)
    


for it, input in enumerate(y[0]):
    answer = y[1][it]
    # forward prop
    a0 = input.reshape(len(input)*len(input[0]))

    
    z1 = np.dot(w1,a0) - b1
    a1 = [sigmoid(x) for x in z1]

    z2 = np.dot(w2,a1) - b2
    a2 = [sigmoid(x) for x in z2]

    z3 = np.dot(w3,a2) - b3
    a3 = [sigmoid(x) for x in z3]
    
    if answer == a3.index(max(a3))+1: correct+=1
    print(correct/(it+1)*100," percent correct", end="\r", flush=True)

print(w3)
print(correct/(it+1)*100," percent correct")
# weights = np.random.randn(9,3)
# weights_2 = np.random.randn(3,2)

# for test in testCases.inputs:
#     print("Test Case: ")
#     input_flatten = np.array(test["testCase"]).reshape(9)
#     print("Input: ", input_flatten)

#     bias = np.random.randn(3)

#     hidden = [sigmoid(x) for x in np.subtract(np.dot(input_flatten,weights),bias)]

#     print("Hidden Layer: ", hidden)

#     bias_2 = np.random.randn(2)
#     z_2 = np.subtract(np.dot(hidden,weights_2),bias_2)
#     hidden_2 = [sigmoid(x) for x in z_2]

#     print("Output Layer: ", hidden_2)
    
#     cost = np.square(np.subtract(hidden_2, np.array(test["answer"])))

#     print("Average Cost: ", np.sum(cost))
#     stuff = np.multiply(2, np.subtract(hidden_2, np.array(test["answer"])))
#     print(stuff)
#     print("sigmoid'(z(L))", [sigmoid_prime(x) for x in z_2])
#     stuff2 = np.multiply(stuff, [sigmoid_prime(x) for x in z_2])
#     print("Stuff 2", stuff2)
#     descent = np.multiply(-1,[sum(np.multiply(hidden,y)) for y in stuff2])
#     print("Descent", descent)

#     weight_changes = [sigmoid(x) for x in descent]
#     print("Weight Changes", weight_changes)



    