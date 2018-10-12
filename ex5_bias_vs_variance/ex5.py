# %% Machine Learning Online Class
# %  Exercise 5 | Regularized Linear Regression and Bias-Variance
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  exercise. You will need to complete the following functions:
# %
# %     linearRegCostFunction.m
# %     learningCurve.m
# %     validationCurve.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#
# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  The following code will load the dataset into your environment and plot
# %  the data.
# %

import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
from ex5_bias_vs_variance.linearRegCostFunction import linearRegCostFunction, linearRegGradient
from ex5_bias_vs_variance.trainLinearReg import trainLinearReg
from ex5_bias_vs_variance.learningCurve import learningCurve
from ex5_bias_vs_variance.polyFeatures import polyFeatures
from ex5_bias_vs_variance.validationCurve import validationCurve
from ex1_linear_regression.featureNormalize import featureNormalize

# % Load Training Data
print('Loading and Visualizing Data ...\n')
#
# % Load from ex5data1:
# % You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = io.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = (
    mat['X'],
    mat['y'],
    mat['Xval'],
    mat['yval'],
    mat['Xtest'],
    mat['ytest']
)

#  m = Number of examples
m = X.shape[0]
#
# % Plot training data
plt.plot(X, y, 'rx', markersize=5, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =========== Part 2: Regularized Linear Regression Cost =============
# %  You should now implement the cost function for regularized linear
# %  regression.
# %
#
theta = np.array((1, 1))
J = linearRegCostFunction(theta, np.hstack((np.ones((m, 1)), X)), y, 1)

print(['Cost at theta = [1 ; 1]: %f '
          '\n(this value should be about 303.993192)\n'], J)
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =========== Part 3: Regularized Linear Regression Gradient =============
# %  You should now implement the gradient for regularized linear
# %  regression.
# %
#
grad = linearRegGradient(theta, np.hstack((np.ones((m, 1)), X)), y, 1)
#
print(['Gradient at theta = [1 ; 1]:  [%f; %f] '
          '\n(this value should be about [-15.303016; 598.250744])\n'],
         grad)
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% =========== Part 4: Train Linear Regression =============
# %  Once you have implemented the cost and gradient correctly, the
# %  trainLinearReg function will use your cost function to train
# %  regularized linear regression.
# %
# %  Write Up Note: The data is non-linear, so this will not give a great
# %                 fit.
# %
#
# %  Train linear regression with lambda = 0
L = 0
theta = trainLinearReg(np.hstack((np.ones((m, 1)), X)), y, L)

# %  Plot fit over the data
plt.scatter(X, y, s=25, c='r', marker='x', linewidth=1.5)
plt.plot(X, np.dot(np.hstack((np.ones((m, 1)), X)), theta), '-', linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
# hold off;
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% =========== Part 5: Learning Curve for Linear Regression =============
# %  Next, you should implement the learningCurve function.
# %
# %  Write Up Note: Since the model is underfitting the data, we expect to
# %                 see a graph with "high bias" -- Figure 3 in ex5.pdf
# %
#
L = 0
error_train, error_val = learningCurve(
    np.hstack((np.ones((m, 1)), X)),
    y,
    np.hstack((np.ones((Xval.shape[0], 1)), Xval)),
    yval,
    L
)
plt.plot(
    np.arange(1, m+1),
    error_train,
    '-b',
    label='Training error',
)
plt.plot(
    np.arange(1, m+1),
    error_val,
    '-g',
    label='Validation error'
)
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis(np.array((1, 13, 1, 150)))
plt.show()
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n', i, error_train[i], error_val[i])
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =========== Part 6: Feature Mapping for Polynomial Regression =============
# %  One solution to this is to use polynomial regression. You should now
# %  complete polyFeatures to map each example into its powers
# %
#
p = 8

# % Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, _, _ = featureNormalize(X_poly)
X_poly = np.hstack((np.ones((m, 1)), X_poly))

# % Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test, _, _ = featureNormalize(X_poly_test)
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

# % Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val, _, _ = featureNormalize(X_poly_val)
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

print('Normalized Training Example 1:\n')
print('  %f  \n', X_poly[0, :])
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
#
# %% =========== Part 7: Learning Curve for Polynomial Regression =============
# %  Now, you will get to experiment with polynomial regression with multiple
# %  values of lambda. The code below runs polynomial regression with
# %  lambda = 0. You should try running the code with different values of
# %  lambda to see how the fit and learning curve change.
# %

L = 1
theta = trainLinearReg(X_poly, y, L)

# figure(2);
error_train, error_val = learningCurve(
    X_poly,
    y,
    X_poly_val,
    yval,
    L
)
plt.plot(
    np.arange(0, m),
    error_train,
    '-b',
    label='Training error',
)
plt.plot(
    np.arange(0, m),
    error_val,
    '-g',
    label='Validation error'
)
plt.title('Polynomial Regression Learning curve (lambda = {})'.format(L))
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis(np.array((0, 13, 0, 100)))
plt.show()

# % Plot training data and fit
# figure(1);
plt.scatter(X, y, s=25, c='r', marker='x', linewidth=1.5)
x = np.arange(np.min(X) - 2, np.max(X) + 5, 0.05)
X_poly_plot = polyFeatures(x, p)
X_poly_plot, _, _ = featureNormalize(X_poly_plot)
X_poly_plot = np.hstack((np.ones((x.shape[0], 1)), X_poly_plot))
plt.plot(x, X_poly_plot.dot(theta), linestyle='--', color='b')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {})'.format(L))
plt.show()


#
# fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
# fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
# for i = 1:m
#     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
# end
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =========== Part 8: Validation for Selecting Lambda =============
# %  You will now implement validationCurve to test various values of
# %  lambda on a validation set. You will then use this to select the
# %  "best" lambda value.
# %
#
lambda_vec, error_train, error_val = validationCurve(
    X_poly,
    y,
    X_poly_val,
    yval
)
#
# close all;
m = lambda_vec.size
plt.plot(
    np.arange(0, m),
    error_train,
    '-b',
    label='Training error',
)
plt.plot(
    np.arange(0, m),
    error_val,
    '-g',
    label='Validation error'
)
plt.title('Validation curve to find best lambda')
plt.legend()
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.axis(np.array((0, 10, 0, 20)))
plt.show()

# fprintf('lambda\t\tTrain Error\tValidation Error\n');
# for i = 1:length(lambda_vec)
# 	fprintf(' %f\t%f\t%f\n', ...
#             lambda_vec(i), error_train(i), error_val(i));
# end
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
