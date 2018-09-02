# %NORMALEQN Computes the closed-form solution to linear regression
# %   NORMALEQN(X,y) computes the closed-form solution to linear
# %   regression using the normal equations.
#
# theta = zeros(size(X, 2), 1);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the code to compute the closed form solution
# %               to linear regression and put the result in theta.
# %
#
# % ---------------------- Sample Solution ----------------------
#
#
#
#
# % -------------------------------------------------------------
#
#
# % ============================================================
#
# end

import numpy as np

def normalEqn(X, y):
    inverse = np.linalg.pinv(np.dot(X.T, X))
    x_times_y = np.dot(X.T, y)
    return np.dot(inverse, x_times_y)

np.set_printoptions(suppress=True)
file = np.loadtxt("ex1data2.txt", delimiter=",")
X = file[..., :-1]
X = np.insert(X, 0, 1, axis=1)
y = file[..., -1]
theta = normalEqn(X, y)
# Prediction of new data: house size 1650 and num of beds 3
new_data = np.array([1, 1650, 3])
prediction = np.dot(theta, new_data)
# 293081.4643349727
print(prediction)
