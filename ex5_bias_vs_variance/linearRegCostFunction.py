# %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
# %regression with multiple variables
# %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
# %   cost of using theta as the parameter for linear regression to fit the
# %   data points in X and y. Returns the cost in J and the gradient in grad
#
# % Initialize some useful values
# m = length(y); % number of training examples
#
# % You need to return the following variables correctly
# J = 0;
# grad = zeros(size(theta));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost and gradient of regularized linear
# %               regression for a particular choice of theta.
# %
# %               You should set J to the cost and grad to the gradient.
# %

import numpy as np
from ex1_linear_regression.computeCost import computeCost

def linearRegCostFunction(theta, X, y, L):
    m = X.shape[0]
    return computeCost(X, y.T, theta) + np.sum(np.square(theta[1:])) * L / (2*m)

def linearRegGradient(theta, X, y, L):
    m = X.shape[0]
    hypothesis = np.dot(X, theta.reshape(-1, 1))
    sum = np.dot(X.T, np.subtract(hypothesis, y))
    return (sum / m) + L / (m * theta[:1].reshape(-1, 1))
