# %% Machine Learning Online Class
# %  Exercise 6 | Support Vector Machines
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  exercise. You will need to complete the following functions:
# %
# %     gaussianKernel.m
# %     dataset3Params.m
# %     processEmail.m
# %     emailFeatures.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#
# %% =============== Part 1: Loading and Visualizing Data ================
# %  We start the exercise by first loading and visualizing the dataset.
# %  The following code will load the dataset into your environment and plot
# %  the data.
# %

import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
# Using my implementation from the logisitic regression ex since plotting the data is similar for linear case
from ex2_logistic_regression.plotData import plotData
from ex6_support_vector_machines.gaussianKernel import gaussianKernel
from ex6_support_vector_machines.visualizeBoundary import visualizeBoundary
from ex6_support_vector_machines.dataset3Params import dataset3Params


print('Loading and Visualizing Data ...\n')

# % Load from ex6data1:
# % You will have X, y in your environment
mat = io.loadmat('ex6data1.mat')
X, y = (
    mat['X'],
    mat['y'],
)

# % Plot training data
plotData(X, y, ("Pos", "Neg"))
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% ==================== Part 2: Training Linear SVM ====================
# %  The following code will train a linear SVM on the dataset and plot the
# %  decision boundary learned.
# %
#
# % Load from ex6data1:
# % You will have X, y in your environment
# load('ex6data1.mat');
#
print('\nTraining Linear SVM ...\n')

# % You should try to change the C value below and see how the decision
# % boundary varies (e.g., try C = 1000)
C = 1
model = svm.SVC(C=C, kernel='linear')
model.fit(X, y.flatten())
coef = model.coef_.ravel()
intercept = model.intercept_.ravel()
xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
yp = -1.0 * (coef[0] * xp + intercept[0]) / coef[1]
plt.plot(xp, yp, linestyle='-', color='b')
plotData(X, y, ("Pos", "Neg"))
plt.show()

#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =============== Part 3: Implementing Gaussian Kernel ===============
# %  You will now implement the Gaussian kernel to use
# %  with the SVM. You should complete the code in gaussianKernel.m
# %
# print('\nEvaluating the Gaussian Kernel ...\n')
#
# x1 = np.array((1, 2, 1))
# x2 = np.array((0, 4, -1))
# sigma = 2
# sim = gaussianKernel(x1, x2, sigma)
#
# print(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :'
#        '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim)
# #
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =============== Part 4: Visualizing Dataset 2 ================
# %  The following code will load the next dataset into your environment and
# %  plot the data.
# %

print('Loading and Visualizing Data ...\n')

# % Load from ex6data2:
# % You will have X, y in your environment
mat = io.loadmat('ex6data2.mat')
X, y = (
    mat['X'],
    mat['y'],
)

# % Plot training data
plotData(X, y, ("Pos", "Neg"))
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# %  After you have implemented the kernel, we can now use it to train the
# %  SVM classifier.
# %
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
#
#
# % SVM Parameters
C = 500
gamma=10

# % We set the tolerance and max_passes lower here so that the code will run
# % faster. However, in practice, you will want to run the training to
# % convergence.
model = svm.SVC(C=C,
                kernel='rbf',
                gamma=gamma
            )
model.fit(X, y.flatten())
visualizeBoundary(X, y, model)
plt.xlim([0, 1])
plt.ylim([0.4, 1])
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =============== Part 6: Visualizing Dataset 3 ================
# %  The following code will load the next dataset into your environment and
# %  plot the data.
# %

print('Loading and Visualizing Data ...\n')

# % Load from ex6data3:
# % You will have X, y in your environment

mat = io.loadmat('ex6data3.mat')
X, y, Xval, yval = (
    mat['X'],
    mat['y'],
    mat['Xval'],
    mat['yval']
)
# % Plot training data
plotData(X, y, ("Pos", "Neg"))
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
#
# %  This is a different dataset that you can use to experiment with. Try
# %  different values of C and sigma here.
#
# % Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)
# % Train the SVM
model = svm.SVC(C=C,
                kernel='rbf',
                gamma=sigma
            )
model.fit(X, y.flatten())
visualizeBoundary(X, y, model)
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
