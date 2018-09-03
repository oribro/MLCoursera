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

def plotData(X, y, labels):
    colors = ('k' if yi else 'y' for yi in y)
    markers = ('+' if yi else 'o' for yi in y)
    for x, y, c, m in zip(X[:, 0], X[:, 1], colors, markers):
        plt.scatter(x, y, c=c, marker=m)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    true_label = mpatches.Patch(color='k', label=labels[0])
    false_label = mpatches.Patch(color='y', label=labels[1])
    plt.legend(handles=[true_label, false_label], loc=1, fontsize='small')
    plt.show()
    return plt


file1 = np.loadtxt("ex2data1.txt", delimiter=",")
X = file1[..., :-1]
y = file1[..., -1]
labels = ('Admitted', 'Not admitted')
plotData(X, y, labels)

file2 = np.loadtxt("ex2data2.txt", delimiter=",")
X = file2[..., :-1]
y = file2[..., -1]
labels = ('y = 1', 'y = 0')
plotData(X, y, labels)