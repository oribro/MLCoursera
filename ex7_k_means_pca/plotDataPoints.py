# %PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
# %index assignments in idx have the same color
# %   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
# %   with the same index assignments in idx have the same color
#
# % Create palette
# palette = hsv(K + 1);
# colors = palette(idx, :);
#
# % Plot the data
# scatter(X(:,1), X(:,2), 15, colors);
#
# end

import matplotlib.pyplot as plt
import colorsys


def plotDataPoints(X, idx, K):
    hsv_tuples = [(x * 1.0 / K, 0.5, 0.5) for x in range(K)]
    rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    colors = [rgb_tuples[index - 1] for index in idx.flatten()]
    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=15,
        c=colors
    )
