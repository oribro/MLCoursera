# %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
# %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
# %   theta as the parameter for regularized logistic regression and the
# %   gradient of the cost w.r.t. to the parameters.
#
# % Initialize some useful values
# m = length(y); % number of training examples
#
# % You need to return the following variables correctly
# J = 0;
# grad = zeros(size(theta));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta.
# %               You should set J to the cost.
# %               Compute the partial derivatives and set grad to the partial
# %               derivatives of the cost w.r.t. each parameter in theta
#
#
#
#
#
#
# % =============================================================
#
# end

import numpy as np
from ex2_logistic_regression.mapFeature import mapFeature
from ex2_logistic_regression.costFunction import costFunction, gradient


def costFunctionReg(theta, X, y, L):
    m = y.size
    return costFunction(theta, X, y) + (np.dot(theta.T[1:], theta[1:]) * (L / (2*m)))

def gradientReg(theta, X, y, L):
    m = y.size
    regularized = gradient(theta, X, y)[1:] + theta[1:] * (L / m)
    return np.insert(regularized, 0, gradient(theta, X, y)[0])

'''
file = np.loadtxt("ex2data2.txt", delimiter=",")
X1 = file[..., 0]
X2 = file[..., 1]
X = mapFeature(X1, X2)
y = file[..., -1]
num_features = np.size(X, 1)
theta = np.zeros(num_features)
np.set_printoptions(suppress=True)
print(costFunctionReg(theta, X, y, 1))
print(gradientReg(theta, X, y, 1)[:5])
test_theta = np.ones(num_features)
print(costFunctionReg(test_theta, X, y, 10))
print(gradientReg(test_theta, X, y, 10)[:5])
'''
