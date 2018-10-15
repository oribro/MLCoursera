# %PLOTPROGRESSKMEANS is a helper function that displays the progress of
# %k-Means as it is running. It is intended for use only with 2D data.
# %   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
# %   points with colors assigned to each centroid. With the previous
# %   centroids, it also plots a line between the previous locations and
# %   current locations of the centroids.
# %
#
# % Plot the examples
# plotDataPoints(X, idx, K);
#
# % Plot the centroids as black x's
# plot(centroids(:,1), centroids(:,2), 'x', ...
#      'MarkerEdgeColor','k', ...
#      'MarkerSize', 10, 'LineWidth', 3);
#
# % Plot the history of the centroids with lines
# for j=1:size(centroids,1)
#     drawLine(centroids(j, :), previous(j, :));
# end
#
# % Title
# title(sprintf('Iteration number %d', i))
#
# end

import matplotlib.pyplot as plt
from ex7_k_means_pca.plotDataPoints import plotDataPoints
from ex7_k_means_pca.drawLine import drawLine

def plotProgresskMeans(X, centroids, previous, idx, K):
    plotDataPoints(X, idx, K)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='k',
        s=30,
        marker='x',
        linewidths=3
    )
    for j in range(centroids.shape[0]):
        drawLine(centroids[j,:], previous[j,:])
