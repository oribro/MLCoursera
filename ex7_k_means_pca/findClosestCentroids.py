# %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
# %   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
# %   in idx for a dataset X where each row is a single example. idx = m x 1
# %   vector of centroid assignments (i.e. each entry in range [1..K])
# %
#
# % Set K
# K = size(centroids, 1);
#
# % You need to return the following variables correctly.
# idx = zeros(size(X,1), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Go over every example, find its closest centroid, and store
# %               the index inside idx at the appropriate location.
# %               Concretely, idx(i) should contain the index of the centroid
# %               closest to example i. Hence, it should be a value in the
# %               range 1..K
# %
# % Note: You can use a for-loop over the examples to compute this.
# %
#
#
#
#
#
#
#
# % =============================================================
#
# end

import numpy as np


def findClosestCentroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros((m, 1))
    for i in range(m):
        # Guarantee intialized distance to be big enough
        min_distance = np.inf
        for j in range(K):
            distance = np.linalg.norm(X[i, :] - centroids[j, :]) ** 2
            if distance < min_distance:
                min_distance = distance
                idx[i] = j + 1

    return idx

