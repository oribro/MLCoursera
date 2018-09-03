# %PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
# %the decision boundary defined by theta
# %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
# %   positive examples and o for the negative examples. X is assumed to be
# %   a either
# %   1) Mx3 matrix, where the first column is an all-ones column for the
# %      intercept.
# %   2) MxN, N>3 matrix, where the first column is all-ones
#
# % Plot Data
# plotData(X(:,2:3), y);
# hold on
#
# if size(X, 2) <= 3
#     % Only need 2 points to define a line, so choose two endpoints
#     plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
#
#     % Calculate the decision boundary line
#     plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
#
#     % Plot, and adjust axes for better viewing
#     plot(plot_x, plot_y)
#
#     % Legend, specific for the exercise
#     legend('Admitted', 'Not admitted', 'Decision Boundary')
#     axis([30, 100, 30, 100])
# else
#     % Here is the grid range
#     u = linspace(-1, 1.5, 50);
#     v = linspace(-1, 1.5, 50);
#
#     z = zeros(length(u), length(v));
#     % Evaluate z = theta*x over the grid
#     for i = 1:length(u)
#         for j = 1:length(v)
#             z(i,j) = mapFeature(u(i), v(j))*theta;
#         end
#     end
#     z = z'; % important to transpose z before calling contour
#
#     % Plot z = 0
#     % Notice you need to specify the range [0, 0]
#     contour(u, v, z, [0, 0], 'LineWidth', 2)
# end
# hold off
#
# end

from ex2_logistic_regression.costFunction import costFunction, gradient
from ex2_logistic_regression.plotData import plotData
from ex2_logistic_regression.sigmoid import sigmoid
from ex2_logistic_regression.predict import predict
# Use Newton Conjugate Gradient to minimize theta
from scipy.optimize import fmin_ncg
import numpy as np
from matplotlib import pyplot as plt

def optimize_J(theta, X, y):
    return fmin_ncg(costFunction, x0=theta, fprime=gradient, args=(X, y), maxiter=400)

def plotDecisionBoundary(theta, X, y):
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
    x1_vec, x2_vec = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    new_X = np.column_stack((np.ones(x1_vec.flatten().size), x1_vec.flatten(), x2_vec.flatten()))
    hypothesis = sigmoid(np.dot(new_X, theta)).reshape(x1_vec.shape)
    plt.contour(x1_vec, x2_vec, hypothesis, levels=0.5, colors='b')
    return

file = np.loadtxt("ex2data1.txt", delimiter=",")
X = file[..., :-1]
y = file[..., -1]
X = np.insert(X, 0, 1, axis=1)
num_features = np.size(X, 1)
theta = np.zeros(num_features)
# Optimization terminated successfully.
#          Current function value: 0.204316
#          Iterations: 25
#          Function evaluations: 63
#          Gradient evaluations: 207
#          Hessian evaluations: 0
# [-22.90706342   0.18820183   0.18323267]
final_theta = optimize_J(theta, X, y)
plotDecisionBoundary(final_theta, X, y)
p = predict(final_theta, X)
# 89
print(np.mean(p == y) * 100)