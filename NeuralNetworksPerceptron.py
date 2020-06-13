# The first Neural Networks code I write on my own :D
import numpy as np

np.random.seed(1)
    
def SigmoidL(x):
    for i in range(len(x)):
        x[i] = 1/(1+np.exp(-x[i]))
    return x

X = [[0, 0, 1],
     [1, 1, 1],
     [1, 0, 1],
     [0, 1, 1]]

y = [0, 1, 1, 0]

weights = np.random.randn(3)*0.10
bias = 0

iters = 20000

print("Starting Weights: ")
print(weights)

for iter in range(iters):
    sit = np.dot(X, weights)+bias
    outputs = SigmoidL(sit)
    error = y - outputs
    adj = np.dot(np.dot(sit, (1 - sit.T)),np.dot(error, X))
    weights += adj
    
print("Weights now: ")
print(weights)
print(Sigmoid(np.dot(X, weights)+bias))


# The code on one input sample that I coded first:

'''X = [1,0,1]

y = 1

w = np.random.randn(3)*0.10

b = 0

print("Weight:")
print(w)
print(np.dot(X,w))

iters = 200

for it in range(iters):
    sit = np.dot(X,w)+b
    out = Sigmoid(sit)
    error = y - out
    adj = np.dot(X,error)*sit*(1-sit)
    w+=adj
    print("In "+str(it))
    print(w)
    print(X*w)'''
