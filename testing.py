import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


a = np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3]
])
b = np.array([4,5,6])

print(a.shape,b.shape)
print(np.e**4/sum(np.e**b))
print(np.e**5/sum(np.e**b))
print(np.e**6/sum(np.e**b))