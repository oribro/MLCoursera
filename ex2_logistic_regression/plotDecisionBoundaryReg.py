from ex2_logistic_regression.costFunctionReg import costFunctionReg, gradientReg
from ex2_logistic_regression.mapFeature import mapFeature
from ex2_logistic_regression.sigmoid import  sigmoid
from ex2_logistic_regression.plotData import plotData
# Use Newton Conjugate Gradient to minimize theta
from scipy.optimize import fmin_ncg
import numpy as np
from matplotlib import pyplot as plt

def optimize_J_reg(theta, X, y, L):
    return fmin_ncg(costFunctionReg, x0=theta, fprime=gradientReg, args=(X, y, L), maxiter=400)

def plotDecisionBoundaryReg(theta, X, y):
    plt = plotData(X, y, ('y = 1', 'y = 0'))
    x1_vec, x2_vec = np.meshgrid(np.linspace(-1, 1.5), np.linspace(-1, 1.5))
    hypothesis = sigmoid(np.dot(mapFeature(x1_vec.flatten(), x2_vec.flatten()), theta)).reshape(x1_vec.shape)
    plt.contour(x1_vec, x2_vec, hypothesis, levels=0.5, colors='g')
    plt.show()
    return

file = np.loadtxt("ex2data2.txt", delimiter=",")
X1 = file[..., 0]
X2 = file[..., 1]
X_reg = mapFeature(X1, X2)
y = file[..., -1]
X = file[..., :-1]
num_features = np.size(X_reg, 1)
theta = np.zeros(num_features)
final_theta = optimize_J_reg(theta, X_reg, y, 1)
print(plotDecisionBoundaryReg(final_theta, X, y))