# %NORMALIZERATINGS Preprocess data by subtracting mean rating for every
# %movie (every row)
# %   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
# %   has a rating of 0 on average, and returns the mean rating in Ymean.
# %
#
# [m, n] = size(Y);
# Ymean = zeros(m, 1);
# Ynorm = zeros(size(Y));
# for i = 1:m
#     idx = find(R(i, :) == 1);
#     Ymean(i) = mean(Y(i, idx));
#     Ynorm(i, idx) = Y(i, idx) - Ymean(i);
# end
#
# end

import numpy as np


def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i, :] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i, :]

    return Ynorm, Ymean
