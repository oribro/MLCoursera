# %NNCOSTFUNCTION Implements the neural network cost function for a two layer
# %neural network which performs classification
# %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
# %   X, y, lambda) computes the cost and gradient of the neural network. The
# %   parameters for the neural network are "unrolled" into the vector
# %   nn_params and need to be converted back into the weight matrices.
# %
# %   The returned parameter grad should be a "unrolled" vector of the
# %   partial derivatives of the neural network.
# %
#
# % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# % for our 2 layer neural network
# Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                  hidden_layer_size, (input_layer_size + 1));
#
# Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                  num_labels, (hidden_layer_size + 1));
#
# % Setup some useful variables
# m = size(X, 1);
#
# % You need to return the following variables correctly
# J = 0;
# Theta1_grad = zeros(size(Theta1));
# Theta2_grad = zeros(size(Theta2));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: You should complete the code by working through the
# %               following parts.
# %
# % Part 1: Feedforward the neural network and return the cost in the
# %         variable J. After implementing Part 1, you can verify that your
# %         cost function computation is correct by verifying the cost
# %         computed in ex4.m
# %
# % Part 2: Implement the backpropagation algorithm to compute the gradients
# %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
# %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# %         Theta2_grad, respectively. After implementing Part 2, you can check
# %         that your implementation is correct by running checkNNGradients
# %
# %         Note: The vector y passed into the function is a vector of labels
# %               containing values from 1..K. You need to map this vector into a
# %               binary vector of 1's and 0's to be used with the neural network
# %               cost function.
# %
# %         Hint: We recommend implementing backpropagation using a for-loop
# %               over the training examples if you are implementing it for the
# %               first time.
# %
# % Part 3: Implement regularization with the cost function and gradients.
# %
# %         Hint: You can implement this around the code for
# %               backpropagation. That is, you can compute the gradients for
# %               the regularization separately and then add them to Theta1_grad
# %               and Theta2_grad from Part 2.
# %
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
#
#
#
#
#
#
#
#
# % -------------------------------------------------------------
#
# % =========================================================================
#
# % Unroll gradients
# grad = [Theta1_grad(:) ; Theta2_grad(:)];
#
#
# end

import numpy as np
from ex2_logistic_regression.sigmoid import sigmoid
from ex4_neural_networks_learning.sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, L):
    m = X.shape[0]
    # Pulling original thetas out of nn_params
    theta1_shape = (hidden_layer_size, (input_layer_size + 1))
    theta1_size = theta1_shape[0] * theta1_shape[1]
    theta2_shape = (num_labels, (hidden_layer_size + 1))
    theta1, theta2 = (nn_params[:theta1_size].reshape(theta1_shape),
                      nn_params[theta1_size:].reshape(theta2_shape))
    # Input layer
    X = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(theta1, X.T).T
    # Hidden layer
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = np.dot(theta2, a2.T)
    # Output layer
    hypothesis = sigmoid(z3).T
    # Vectorize y for neural network
    new_y = np.zeros((m, num_labels))
    for i in range(m):
        new_y[i, y[i] - 1] = 1
    product = np.multiply(-new_y, np.log(hypothesis)) - np.multiply(1 - new_y, np.log(1 - hypothesis))
    J = np.sum(product) / m
    J_reg = (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:]))) * (L / (2*m))
    J_total = J + J_reg
    # Performing backpropagation
    d3 = hypothesis - new_y
    z2_deriv = sigmoidGradient(z2)
    d2 = np.dot(d3, theta2[:, 1:]) * z2_deriv
    delta1 = np.dot(d2.T, X)
    delta2 = np.dot(d3.T, a2)
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    # Regularization of the gradient
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + L / m * theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + L / m * theta2[:, 1:]
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return (J_total, grad)

