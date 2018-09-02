# %PLOTDATA Plots the data points X and y into a new figure
# %   PLOTDATA(x,y) plots the data points with + for the positive examples
# %   and o for the negative examples. X is assumed to be a Mx2 matrix.
#
# % Create New Figure
# figure; hold on;
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Plot the positive and negative examples on a
# %               2D plot, using the option 'k+' for the positive
# %               examples and 'ko' for the negative examples.
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
# % =========================================================================
#
#
#
# hold off;
#
# end

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def plotData(X, y):
    colors = ('k' if yi else 'y' for yi in y)
    markers = ('+' if yi else 'o' for yi in y)
    for x, y, c, m in zip(X[:, 0], X[:, 1], colors, markers):
        plt.scatter(x, y, c=c, marker=m)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    admitted = mpatches.Patch(color='k', label='Admitted')
    not_admitted = mpatches.Patch(color='y', label='Not admitted')
    plt.legend(handles=[admitted, not_admitted], loc=1, fontsize='small')
    plt.show()
    return


file = np.loadtxt("ex2data1.txt", delimiter=",")
X = file[..., :-1]
y = file[..., -1]
plotData(X, y)