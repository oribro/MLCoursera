# %VISUALIZEFIT Visualize the dataset and its estimated distribution.
# %   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
# %   probability density function of the Gaussian distribution. Each example
# %   has a location (x1, x2) that depends on its feature values.
# %
#
# [X1,X2] = meshgrid(0:.5:35);
# Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
# Z = reshape(Z,size(X1));
#
# plot(X(:, 1), X(:, 2),'bx');
# hold on;
# % Do not plot if there are infinities
# if (sum(isinf(Z)) == 0)
#     contour(X1, X2, Z, 10.^(-20:3:0)');
# end
# hold off;
#
# end

import numpy as np
import matplotlib.pyplot as plt
from ex8_anomaly_detect_recommender_systems.multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    xx = yy = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(xx, yy)
    Z = multivariateGaussian(np.vstack((X1.flatten(), X2.flatten())).T, mu, sigma2).reshape(X1.shape)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c='b',
        marker='x',
        s=10
    )
    if np.sum(np.isinf(Z) == 0):
        plt.contour(
            X1,
            X2,
            Z,
            10.0 ** np.arange(-20, 0, 3)
        )

