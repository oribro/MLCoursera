from ex2_logistic_regression.costFunctionReg import costFunctionReg, gradientReg
from ex2_logistic_regression.mapFeature import mapFeature
# Use Newton Conjugate Gradient to minimize theta
from scipy.optimize import fmin_ncg
import numpy as np
from matplotlib import pyplot as plt

def optimize_J_reg(theta, X, y, L):
    return fmin_ncg(costFunctionReg, x0=theta, fprime=gradientReg, args=(X, y, L), maxiter=400)



file = np.loadtxt("ex2data2.txt", delimiter=",")
X1 = file[..., 0]
X2 = file[..., 1]
X = mapFeature(X1, X2)
y = file[..., -1]
num_features = np.size(X, 1)
theta = np.zeros(num_features)
print(optimize_J_reg(theta, X, y, 0))