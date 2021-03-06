# %% Machine Learning Online Class - Exercise 4 Neural Network Learning
#
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear exercise. You will need to complete the following functions
# %  in this exericse:
# %
# %     sigmoidGradient.m
# %     randInitializeWeights.m
# %     nnCostFunction.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#
# %% Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  You will be working with a dataset that contains handwritten digits.
# %

import scipy.io as io
import numpy as np
from ex3_neural_networks.displayData import displayData
from ex4_neural_networks_learning.nnCostFunction import nnCostFunction
from ex4_neural_networks_learning.sigmoidGradient import sigmoidGradient
from ex4_neural_networks_learning.randInitializeWeights import randInitializeWeights
from ex4_neural_networks_learning.checkNNGradients import checkNNGradients
from ex4_neural_networks_learning.predict import predict
from scipy.optimize import minimize

# % Load Training Data
print('Loading and Visualizing Data ...\n')

mat = io.loadmat('ex4data1.mat')
X = mat['X']
y = mat['y']
displayData(X)


#
# %% ================ Part 2: Loading Parameters ================
# % In this part of the exercise, we load some pre-initialized
# % neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# % Load the weights into variables Theta1 and Theta2
weights = io.loadmat('ex4weights.mat')
theta1, theta2 = (weights['Theta1'], weights['Theta2'])
# % Unroll parameters
nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))

# Test pulling original thetas out of nn_params
# test1, test2 = (nn_params[:theta1.size].reshape(theta1.shape), nn_params[theta1.size:].reshape(theta2.shape))
# print(np.array_equal(test1, theta1))
# print(np.array_equal(test2, theta2))

# %% ================ Part 3: Compute Cost (Feedforward) ================
# %  To the neural network, you should first start by implementing the
# %  feedforward part of the neural network that returns the cost only. You
# %  should complete the code in nnCostFunction.m to return cost. After
# %  implementing the feedforward to compute the cost, you can verify that
# %  your implementation is correct by verifying that you get the same cost
# %  as us for the fixed debugging parameters.
# %
# %  We suggest implementing the feedforward cost *without* regularization
# %  first so that it will be easier for you to debug. Later, in part 4, you
# %  will get to implement the regularized cost.
# %
print('\nFeedforward Using Neural Network ...\n')
#
# % Weight regularization parameter (we set this to 0 here).
L = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, L)

print(['Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n'], J)

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% =============== Part 4: Implement Regularization ===============
# %  Once your cost function implementation is correct, you should now
# %  continue to implement the regularization with the cost.
# %
#
print('\nChecking Cost Function (w/ Regularization) ... \n')
#
# % Weight regularization parameter (we set this to 1 here).
L = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                    num_labels, X, y, L)
#
print(['Cost at parameters (loaded from ex4weights): %f '
          '\n(this value should be about 0.383770)\n'], J)

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% ================ Part 5: Sigmoid Gradient  ================
# %  Before you start implementing the neural network, you will first
# %  implement the gradient for the sigmoid function. You should complete the
# %  code in the sigmoidGradient.m file.
# %

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array((-1, -0.5, 0, 0.5, 1)))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print('%f ', g)
print('\n\n')

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% ================ Part 6: Initializing Pameters ================
# %  In this part of the exercise, you will be starting to implment a two
# %  layer neural network that classifies digits. You will start by
# %  implementing a function to initialize the weights of the neural network
# %  (randInitializeWeights.m)
#
print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# % Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))


#
# %% =============== Part 7: Implement Backpropagation ===============
# %  Once your cost matches up with ours, you should proceed to implement the
# %  backpropagation algorithm for the neural network. You should add to the
# %  code you've written in nnCostFunction.m to return the partial
# %  derivatives of the parameters.

print('\nChecking Backpropagation... \n');

# %  Check gradients by running checkNNGradients
# checkNNGradients(0)
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% =============== Part 8: Implement Regularization ===============
# %  Once your backpropagation implementation is correct, you should now
# %  continue to implement the regularization with the cost and gradient.
# %
#
print('\nChecking Backpropagation (w/ Regularization) ... \n')
#
# %  Check gradients by running checkNNGradients
L = 3
# checkNNGradients(L)

# % Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, L)[0]
#
print(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f '
          '\n(for lambda = 3, this value should be about 0.576051)\n\n'], L, debug_J)
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% =================== Part 8: Training NN ===================
# %  You have now implemented all the code necessary to train a neural
# %  network. To train your neural network, we will now use "fmincg", which
# %  is a function which works similarly to "fminunc". Recall that these
# %  advanced optimizers are able to train our cost functions efficiently as
# %  long as we provide them with the gradient computations.
# %
print('\nTraining Neural Network... \n')
#
# %  After you have completed the assignment, change the MaxIter to a larger
# %  value to see how more training helps.
#
# %  You should also try different values of lambda
L = 0.1

result = minimize(fun=nnCostFunction,
                  x0=nn_params,
                        args=(
                          input_layer_size,
                          hidden_layer_size,
                          num_labels,
                          X,
                          y,
                          L
                      ),
                  method='TNC',
                  jac=True,
                  options={'maxiter': 200}
                )

print(result)
# % Obtain Theta1 and Theta2 back from result

final_theta1 = np.reshape(result.x[0:(hidden_layer_size * (input_layer_size + 1)), ],
                             (hidden_layer_size, input_layer_size + 1))
final_theta2 = np.reshape(result.x[(hidden_layer_size * (input_layer_size + 1)):, ],
                             (num_labels, hidden_layer_size + 1))


# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% ================= Part 9: Visualize Weights =================
# %  You can now "visualize" what the neural network is learning by
# %  displaying the hidden units to see what features they are capturing in
# %  the data.
#
print('\nVisualizing Neural Network... \n')
#
displayData(final_theta1[:, 1:])
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ================= Part 10: Implement Predict =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.
#
pred = predict(final_theta1, final_theta2, X)

'''
Obtained 100% accuracy and J = 0.063 with overfitting: lambda = 0.1, and 200 iterations. Nice!
TODO: Test on new examples
'''

print('\nTraining Set Accuracy: %f\n', np.mean((pred == y.flatten()).astype(float)) * 100)

