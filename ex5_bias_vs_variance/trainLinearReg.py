# %TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
# %regularization parameter lambda
# %   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
# %   the dataset (X, y) and regularization parameter lambda. Returns the
# %   trained parameters theta.
# %
#
# % Initialize Theta
# initial_theta = zeros(size(X, 2), 1);
#
# % Create "short hand" for the cost function to be minimized
# costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
#
# % Now, costFunction is a function that takes in only one argument
# options = optimset('MaxIter', 200, 'GradObj', 'on');
#
# % Minimize using fmincg
# theta = fmincg(costFunction, initial_theta, options);
#
# end

import numpy as np
from scipy.optimize import minimize
from ex5_bias_vs_variance.linearRegCostFunction import linearRegCostFunction, linearRegGradient

def trainLinearReg(X, y, L):
    initial_theta = np.ones((X.shape[1], 1))
    result = minimize(fun=linearRegCostFunction,
                      x0=initial_theta,
                      args=(
                          X,
                          y,
                          L
                      ),
                      method='TNC',
                      jac=linearRegGradient,
                      options={'maxiter': 200}
                    )
    return result.x

