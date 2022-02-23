import numpy as np
from scipy.misc import derivative


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(array, derivative=False):
    # Numerically stable with large exponentials
    n = np.e**(array)
    d = sum(n)
    if derivative:
        return n
    return n / d

a = np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3]
])
b = np.array([4,5,6])

print(a.shape,b.shape)
print(softmax(b,derivative=True))