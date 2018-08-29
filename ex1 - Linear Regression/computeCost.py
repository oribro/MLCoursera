# %COMPUTECOST Compute cost for linear regression
# %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
# %   parameter for linear regression to fit the data points in X and y
#
# % Initialize some useful values
# m = length(y); % number of training examples
#
# % You need to return the following variables correctly
# J = 0;
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta
# %               You should set J to the cost.
#
#
#
#
#
# % =========================================================================
#
# end

import numpy as np

def computeCost(X, y, theta):
    m = y.size
    hypothesis = np.dot(X, theta)
    error_values = np.square(np.subtract(hypothesis, y))
    sum = np.sum(error_values)
    return sum / (2*m)


file = np.loadtxt("ex1data1.txt", delimiter=",")
X = file[(..., 0)]
X = np.expand_dims(X, axis=1)
X = np.insert(X, 0, 1, axis=1)
y = file[(..., 1)]
theta = np.zeros(2)
print(computeCost(X, y, theta))
theta[0] = -1
theta[1] = 2
print(computeCost(X, y, theta))