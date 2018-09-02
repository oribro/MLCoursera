# %SIGMOID Compute sigmoid function
# %   g = SIGMOID(z) computes the sigmoid of z.
#
# % You need to return the following variables correctly
# g = zeros(size(z));
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
# %               vector or scalar).
#
#
#
#
#
# % =============================================================
#
# end

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.arange(-100, 100).reshape(20, 10)
y = sigmoid(x)
plt.plot(x, y)
