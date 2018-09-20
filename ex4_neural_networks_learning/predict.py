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
# h1 = sigmoid([ones(m, 1) X] * Theta1');
# h2 = sigmoid([ones(m, 1) h1] * Theta2');
# [dummy, p] = max(h2, [], 2);
#
# % =========================================================================
#
#
# end

import numpy as np
from ex2_logistic_regression.sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    h1 = sigmoid((np.dot(Theta1, np.hstack((np.ones((m, 1)), X)).T))).T
    h2 = sigmoid((np.dot(Theta2, np.hstack((np.ones((m, 1)), h1)).T))).T
    return np.argmax(sigmoid(h2), axis=1) + 1
