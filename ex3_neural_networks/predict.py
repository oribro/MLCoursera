# %PREDICT Predict the label of an input given a trained neural network
# %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
# %   trained weights of a neural network (Theta1, Theta2)
#
# % Useful values
# m = size(X, 1);
# num_labels = size(Theta2, 1);
#
# % You need to return the following variables correctly
# p = zeros(size(X, 1), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the following code to make predictions using
# %               your learned neural network. You should set p to a
# %               vector containing labels between 1 to num_labels.
# %
# % Hint: The max function might come in useful. In particular, the max
# %       function can also return the index of the max element, for more
# %       information see 'help max'. If your examples are in rows, then, you
# %       can use max(A, [], 2) to obtain the max for each row.
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
# % =========================================================================
#
#
# end

import numpy as np
import scipy.io as io
from ex2_logistic_regression.sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    # Input layer
    X = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(Theta1, X.T).T
    # Hidden layer
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = np.dot(Theta2, a2.T)
    # Output layer
    hypothesis = sigmoid(z3).T
    p = np.argmax(hypothesis, axis=1)
    return p+1


mat = io.loadmat('ex3data1.mat')
weights = io.loadmat('ex3weights.mat')
theta1, theta2 = (weights['Theta1'], weights['Theta2'])
X = mat['X']
y = mat['y']
p = predict(theta1, theta2, X)
# 97.52
print(np.mean((p == y.flatten()).astype(float)) * 100)