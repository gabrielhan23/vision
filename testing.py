import numpy as np

a = np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3]
])
b = np.array([4,5,6])

print(a.shape,b.shape)
print(sum((a*b).T))