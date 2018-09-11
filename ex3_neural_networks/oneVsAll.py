# %ONEVSALL trains multiple logistic regression classifiers and returns all
# %the classifiers in a matrix all_theta, where the i-th row of all_theta
# %corresponds to the classifier for label i
# %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
# %   logistic regression classifiers and returns each of these classifiers
# %   in a matrix all_theta, where the i-th row of all_theta corresponds
# %   to the classifier for label i
#
# % Some useful variables
# m = size(X, 1);
# n = size(X, 2);
#
# % You need to return the following variables correctly
# all_theta = zeros(num_labels, n + 1);
#
# % Add ones to the X data matrix
# X = [ones(m, 1) X];
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: You should complete the following code to train num_labels
# %               logistic regression classifiers with regularization
# %               parameter lambda.
# %
# % Hint: theta(:) will return a column vector.
# %
# % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
# %       whether the ground truth is true/false for this class.
# %
# % Note: For this assignment, we recommend using fmincg to optimize the cost
# %       function. It is okay to use a for-loop (for c = 1:num_labels) to
# %       loop over the different classes.
# %
# %       fmincg works similarly to fminunc, but is more efficient when we
# %       are dealing with large number of parameters.
# %
# % Example Code for fmincg:
# %
# %     % Set Initial theta
# %     initial_theta = zeros(n + 1, 1);
# %
# %     % Set options for fminunc
# %     options = optimset('GradObj', 'on', 'MaxIter', 50);
# %
# %     % Run fmincg to obtain the optimal theta
# %     % This function will return theta and the cost
# %     [theta] = ...
# %         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
# %                 initial_theta, options);
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
# % =========================================================================
#
#
# end

import numpy as np
from ex3_neural_networks.lrCostFunction import lrCostFunction, lrGradient
from scipy.optimize import minimize

def oneVsAll(X, y, num_labels, L):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n+1))
    X = np.hstack((np.ones((m, 1)), X))
    for i in range(1, num_labels+1):
        y_i = ((y == i).astype(int)).flatten()
        # Used optimize.minimize here since fmin_ncg took too long to compute
        all_theta[i-1] = minimize(lrCostFunction, np.zeros((n+1, 1)), args=(X, y_i, L), method=None,
                 jac=lrGradient).x
    return all_theta


