# % MAPFEATURE Feature mapping function to polynomial features
# %
# %   MAPFEATURE(X1, X2) maps the two input features
# %   to quadratic features used in the regularization exercise.
# %
# %   Returns a new feature array with more features, comprising of
# %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
# %
# %   Inputs X1, X2 must be the same size
# %

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def mapFeature(X1, X2):
    X = np.column_stack((X1, X2))
    return PolynomialFeatures(degree=6).fit_transform(X)


