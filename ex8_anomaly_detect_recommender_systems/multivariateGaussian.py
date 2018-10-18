# %MULTIVARIATEGAUSSIAN Computes the probability density function of the
# %multivariate gaussian distribution.
# %    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
# %    density function of the examples X under the multivariate gaussian
# %    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
# %    treated as the covariance matrix. If Sigma2 is a vector, it is treated
# %    as the \sigma^2 values of the variances in each dimension (a diagonal
# %    covariance matrix)
# %
#
# k = length(mu);
#
# if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
#     Sigma2 = diag(Sigma2);
# end
#
# X = bsxfun(@minus, X, mu(:)');
# p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
#     exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
#
# end

import numpy as np


def multivariateGaussian(X, mu, sigma2):
    k = len(mu)
    if len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)
    X_mu = X - mu
    p = (2 * np.pi) ** (-k / 2.0) * np.linalg.det(sigma2) ** (-0.5) * np.exp(
        -0.5 * np.sum(
            X_mu.dot(np.linalg.pinv(sigma2)) * X_mu,
            axis=1
        )
    )
    return p