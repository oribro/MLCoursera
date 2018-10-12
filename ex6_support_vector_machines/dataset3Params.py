# %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
# %where you select the optimal (C, sigma) learning parameters to use for SVM
# %with RBF kernel
# %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
# %   sigma. You should complete this function to return the optimal C and
# %   sigma based on a cross-validation set.
# %
#
# % You need to return the following variables correctly.
# C = 1;
# sigma = 0.3;
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Fill in this function to return the optimal C and sigma
# %               learning parameters found using the cross validation set.
# %               You can use svmPredict to predict the labels on the cross
# %               validation set. For example,
# %                   predictions = svmPredict(model, Xval);
# %               will return the predictions on the cross validation set.
# %
# %  Note: You can compute the prediction error using
# %        mean(double(predictions ~= yval))
# %
#

from sklearn import svm
import numpy as np

def get_possible_params(C_base, sigma_base, num_of_multiplications, mult_factor):
    C_constants = [C_base * (mult_factor ** i) for i in range(num_of_multiplications)]
    sigmas = [sigma_base * (mult_factor ** i) for i in range(num_of_multiplications)]
    return C_constants, sigmas

def dataset3Params(X, y, Xval, yval):
    C = 0.1
    sigma = 0.01
    num_of_multiplications = 12
    mult_factor = 2
    C_constants, sigmas = get_possible_params(C,
                                              sigma,
                                              num_of_multiplications,
                                              mult_factor
                                              )
    # Matrix with prediction errors using all possible (C, sigma) pairs
    # Rows in the matrix are C and columns are sigma
    tests = np.zeros((num_of_multiplications, num_of_multiplications))
    for i in range(num_of_multiplications):
        for j in range(num_of_multiplications):
            C = C_constants[i]
            sigma = sigmas[j]
            model = svm.SVC(C=C,
                            kernel='rbf',
                            gamma=sigma
                            )
            model.fit(X, y.flatten())
            tests[i, j] = model.score(Xval, yval)

    max_score_indices = np.unravel_index(tests.argmax(), tests.shape)
    return C_constants[max_score_indices[0]], sigmas[max_score_indices[1]]
