# %CHECKCOSTFUNCTION Creates a collaborative filering problem
# %to check your cost function and gradients
# %   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem
# %   to check your cost function and gradients, it will output the
# %   analytical gradients produced by your code and the numerical gradients
# %   (computed using computeNumericalGradient). These two gradient
# %   computations should result in very similar values.
#
# % Set lambda
# if ~exist('lambda', 'var') || isempty(lambda)
#     lambda = 0;
# end
#
# %% Create small problem
# X_t = rand(4, 3);
# Theta_t = rand(5, 3);
#
# % Zap out most entries
# Y = X_t * Theta_t';
# Y(rand(size(Y)) > 0.5) = 0;
# R = zeros(size(Y));
# R(Y ~= 0) = 1;
#
# %% Run Gradient Checking
# X = randn(size(X_t));
# Theta = randn(size(Theta_t));
# num_users = size(Y, 2);
# num_movies = size(Y, 1);
# num_features = size(Theta_t, 2);
#
# numgrad = computeNumericalGradient( ...
#                 @(t) cofiCostFunc(t, Y, R, num_users, num_movies, ...
#                                 num_features, lambda), [X(:); Theta(:)]);
#
# [cost, grad] = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, ...
#                           num_movies, num_features, lambda);
#
# disp([numgrad grad]);
# fprintf(['The above two columns you get should be very similar.\n' ...
#          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
#
# diff = norm(numgrad-grad)/norm(numgrad+grad);
# fprintf(['If your cost function implementation is correct, then \n' ...
#          'the relative difference will be small (less than 1e-9). \n' ...
#          '\nRelative Difference: %g\n'], diff);
#
# end

import numpy as np
from ex4_neural_networks_learning.computeNumericalGradient import computeNumericalGradient
from ex8_anomaly_detect_recommender_systems.cofiCostFunc import cofiCostFunc


def checkCostFunction(L = 0):
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    Y = np.dot(X_t, Theta_t.T)
    Y[Y > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users= Y.shape
    num_features = Theta_t.shape[1]
    costFunc = lambda p: cofiCostFunc(
        p,
        Y,
        R,
        num_users,
        num_movies,
        num_features,
        L
    )
    params = np.concatenate((X.flatten(), Theta.flatten()))
    cost, grad = costFunc(params)
    numgrad = computeNumericalGradient(costFunc, params)

    np.set_printoptions(precision=12, suppress=True)
    print((numgrad, grad))

    print('The above two columns you get should be very similar.\n'
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    assert diff < 10 ** -9

    print('If your cost function implementation is correct, then \n'
           'the relative difference will be small (less than 1e-9). \n'
             '\nRelative Difference: {}\n'.format(diff))



