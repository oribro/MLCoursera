
# %GRADIENTDESCENT Performs gradient descent to learn theta
# %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
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
#     %       of the cost function (computeCost) and gradient here.
#     %
#
#
#
#
#
#
#     % ===#
# =========================================================
#
#     % Save the cost J in every iteration
#     J_history(iter) = computeCost(X, y, theta);
#
# end
#
# end

import numpy as np
from matplotlib import pyplot as plt

def computeCost(X, y, theta):
    m = y.size
    hypothesis = np.dot(X, theta)
    error_values = np.square(np.subtract(hypothesis, y))
    sum = np.sum(error_values)
    return sum / (2*m)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        sum = np.dot(X.T, np.subtract(hypothesis, y))
        J_history[i] = computeCost(X, y, theta)
        theta -= (alpha / m) * sum
    return (theta, J_history)

def plotHypothesis(x, y, theta):
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    hypothesis = theta[0] + theta[1] * x
    plt.scatter(x[:, 1], y, s=30, c='r', marker='x', label='Training data')
    plt.plot(x, hypothesis, c='b', label='Linear Regression')
    plt.legend(loc=4);
    plt.show()

iterations = 1500;
alpha = 0.01;
file = np.loadtxt("ex1data1.txt", delimiter=",")
X = file[(..., 0)]
X = np.expand_dims(X, axis=1)
X = np.insert(X, 0, 1, axis=1)
y = file[(..., 1)]
theta = np.zeros(2)
theta, J = gradientDescent(X, y, theta, alpha, iterations)
# Predictions
print([1,3.5] * theta * 10000)
print([1, 7] * theta * 10000)
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.plot(J)
plt.show()
# Hypothesis function
plotHypothesis(X, y, theta)