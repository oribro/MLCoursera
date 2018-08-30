# %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
# %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
# %   taking num_iters gradient steps with learning rate alpha
#
# % Initialize some useful values
# m = length(y); % number of training examples
# J_history = zeros(num_iters, 1);
#
# for iter = 1:num_iters
#
#     % ====================== YOUR CODE HERE ======================
#     % Instructions: Perform a single gradient step on the parameter vector
#     %               theta.
#     %
#     % Hint: While debugging, it can be useful to print out the values
#     %       of the cost function (computeCostMulti) and gradient here.
#     %
#
#
#
#
#
#
#
#
#
#
#
#     % ============================================================
#
#     % Save the cost J in every iteration
#     J_history(iter) = computeCostMulti(X, y, theta);
#
# end
#
# end

import numpy as np
from matplotlib import pyplot as plt

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

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        sum = np.dot(X.T, np.subtract(hypothesis, y))
        J_history[i] = computeCostMulti(X, y, theta)
        theta -= (alpha / m) * sum
    return (theta, J_history)


iterations = 400;
alpha = 0.01;
file = np.loadtxt("ex1data2.txt", delimiter=",")
X = file[..., :-1]
y = file[..., -1]
X_norm, mu, sigma = featureNormalize(X)
X = np.insert(X_norm, 0, 1, axis=1)
num_features = np.size(X, 1)
theta = np.zeros(num_features)
theta, J = gradientDescentMulti(X, y, theta, alpha, iterations)
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.plot(J)
plt.show()

