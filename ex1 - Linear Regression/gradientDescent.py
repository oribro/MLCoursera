
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
#
#     % ============================================================
#
#     % Save the cost J in every iteration
#     J_history(iter) = computeCost(X, y, theta);
#
# end
#
# end

import numpy as np
from . import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        diff = np.subtract(hypothesis, y)
        sum = np.sum(diff)
        J_history[i] = computeCost(X, y, theta)


iterations = 1500;
alpha = 0.01;
file = np.loadtxt("ex1data1.txt", delimiter=",")
X = file[(..., 0)]
X = np.expand_dims(X, axis=1)
X = np.insert(X, 0, 1, axis=1)
y = file[(..., 1)]
theta = np.zeros(2)
print(gradientDescent(X, y, theta, alpha, iterations))