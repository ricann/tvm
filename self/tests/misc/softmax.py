import numpy as np
import math

def softmax(x):
    print(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax([100,200,300]))
print(softmax([10,20,30]))
print(softmax([1,2,3]))
print(softmax([0.1,0.2,0.3]))
print(softmax([0.01,0.02,0.03]))
