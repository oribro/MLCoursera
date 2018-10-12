# function visualizeBoundary(X, y, model, varargin)
# %VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
# %   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
# %   boundary learned by the SVM and overlays the data on it
#
# % Plot the training data on top of the boundary
# plotData(X, y)
#
# % Make classification predictions over a grid of values
# x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
# x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
# [X1, X2] = meshgrid(x1plot, x2plot);
# vals = zeros(size(X1));
# for i = 1:size(X1, 2)
#    this_X = [X1(:, i), X2(:, i)];
#    vals(:, i) = svmPredict(model, this_X);
# end
#
# % Plot the SVM boundary
# hold on
# contour(X1, X2, vals, [0.5 0.5], 'b');
# hold off;
#
# end

import numpy as np
import matplotlib.pyplot as plt
from ex2_logistic_regression.plotData import plotData

def visualizeBoundary(X, y, model):
   plotData(X, y, ("Pos", "Neg"))
   # Make classification predictions over a grid of values
   x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
   x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
   X1, X2 = np.meshgrid(x1plot, x2plot)
   vals = np.zeros(X1.shape)
   for i in range(X1.shape[1]):
      this_X = np.hstack((X1[:, i:i + 1], X2[:, i:i + 1]))
      vals[:, i] = model.predict(this_X)

   # Plot the SVM boundary
   plt.contour(X1, X2, vals, levels=[0])