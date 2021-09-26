import numpy
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Loading the data set
data_x = numpy.array([
    [1, 20, 20],
    [1, 16, 16],
    [1, 19.8, 20],
    [1, 18.4, 18],
    [1, 17.1, 17],
    [1, 15.5, 16],
    [1, 14.7, 15],
    [1, 17.1, 17],
    [1, 15.4, 15],
    [1, 16.2, 16],
    [1, 15, 15],
    [1, 17.2, 17],
    [1, 16, 16],
    [1, 17, 17]
])

data_y = numpy.array([
    [88.6],
    [71.6],
    [93.3],
    [84.3],
    [80.6],
    [75.2],
    [69.7],
    [82],
    [69.4],
    [83.3],
    [79.6],
    [82.6],
    [80.6],
    [83.5],
])

X = data_x[:]
y = data_y[:]

# Compute the pseudo-inverse and return w
w = inv(X.T.dot(X)).dot(X.T).dot(y)
print(f"1-1-1. The optimal value for the weight vector: {w}")
print(f"1-1-2. h(x) = {w[0]} + {w[1]}x1 + {w[2]}x2")

yhat = X.dot(w)

x1_1 = 14.4
x1_2 = 17
yhat1 = w[0] + w[1] * x1_1 + w[2] * x1_2
print(f"1-2. when x=[14.4, 17], y={yhat1}")
x2_1 = 18
x2_2 = 17
yhat2 = w[0] + w[1] * x2_1 + w[2] * x2_2
print(f"1-2. when x=[18, 17], y={yhat2}")

# Illustration of linear regression
plt.title(f"Multiple Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(w[1] * X[:, 1:2] + w[2] * X[:, 2:3], y)
plt.plot(w[1] * X[:, 1:2] + w[2] * X[:, 2:3], yhat)
plt.show()
