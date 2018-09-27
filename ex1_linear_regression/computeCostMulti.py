# %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
# %   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
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

def featureNormalize(X):
    num_features = np.size(X, 1)
    X_norm = X
    mu = np.zeros(num_features)
    sigma = np.zeros(num_features)
    for i in range(num_features):
        mu[i] = X[:, i].mean()
        sigma[i] = X[:, i].std()
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return (X_norm, mu, sigma)

def computeCostMulti(X, y, theta):
    m = y.size
    hypothesis = np.dot(theta, X.T)
    error_values = hypothesis - y

    return (np.dot(error_values, error_values.T)) / (2*m)


# np.set_printoptions(suppress=True)
# file = np.loadtxt("ex1data2.txt", delimiter=",")
# X = file[..., :-1]
# y = file[..., -1]
# X_norm, mu, sigma = featureNormalize(X)
# X = np.insert(X_norm, 0, 1, axis=1)
# num_features = np.size(X, 1)
# theta = np.zeros(num_features)
# print(computeCostMulti(X, y, theta))
