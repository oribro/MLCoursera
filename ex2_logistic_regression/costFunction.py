# %COSTFUNCTION Compute cost and gradient for logistic regression
# %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
# %   parameter for logistic regression and the gradient of the cost
# %   w.r.t. to the parameters.
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
# %
# % Note: grad should have the same dimensions as theta
# %
#
#
#
#
#
#
#
#
# % =============================================================
#
# end

from ex2_logistic_regression.sigmoid import sigmoid
import numpy as np

# Built-in optimization function fmin_ncg requires to seperate cost function from the gradient
# See plotDecisionBoundary for details
def costFunction(theta, X, y):
    m = y.size
    hypothesis = sigmoid(np.dot(theta, X.T))
    return (np.dot(-y, np.log(hypothesis)) - np.dot(1 - y, np.log(1 - hypothesis))) / m

def gradient(theta, X, y):
    m = y.size
    hypothesis = sigmoid(np.dot(theta, X.T))
    return (np.dot(X.T, hypothesis - y)) / m



# file = np.loadtxt("ex2data1.txt", delimiter=",")
# X = file[..., :-1]
# y = file[..., -1]
# X = np.insert(X, 0, 1, axis=1)
# num_features = np.size(X, 1)
# theta = np.zeros(num_features)
# print(costFunction(theta, X , y))
# test_theta = np.array([-24, 0.2, 0.2])
