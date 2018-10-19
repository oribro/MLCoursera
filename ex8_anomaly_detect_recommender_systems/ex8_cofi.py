# %% Machine Learning Online Class
# %  Exercise 8 | Anomaly Detection and Collaborative Filtering
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  exercise. You will need to complete the following functions:
# %
# %     estimateGaussian.m
# %     selectThreshold.m
# %     cofiCostFunc.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %

import scipy.io as io
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
from ex8_anomaly_detect_recommender_systems.cofiCostFunc import cofiCostFunc
from ex8_anomaly_detect_recommender_systems.checkCostFunction import checkCostFunction
from ex8_anomaly_detect_recommender_systems.loadMovieList import loadMovieList
from ex8_anomaly_detect_recommender_systems.normalizeRatings import normalizeRatings

# %% =============== Part 1: Loading movie ratings dataset ================
# %  You will start by loading the movie ratings dataset to understand the
# %  structure of the data.
# %
print('Loading movie ratings dataset.\n\n')

# %  Load data
mat = io.loadmat('ex8_movies.mat')
R, Y = (
    mat['R'],
    mat['Y']
)
# %  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {} / 5\n\n'.format(np.mean(Y[0, R[0, :] == 1])))

# %  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ============ Part 2: Collaborative Filtering Cost Function ===========
# %  You will now implement the cost function for collaborative filtering.
# %  To help you debug your cost function, we have included set of weights
# %  that we trained on that. Specifically, you should complete the code in
# %  cofiCostFunc.m to return J.

# %  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
mat = io.loadmat('ex8_movieParams.mat')
X, Theta, num_users, num_movies = (
    mat['X'],
    mat['Theta'],
    mat['num_users'],
    mat['num_movies']
)
# %  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]
params = np.concatenate((X.flatten(), Theta.flatten()))

# %  Evaluate cost function
J = cofiCostFunc(
    params,
    Y,
    R,
    num_users,
    num_movies,
    num_features,
    0
)

print('Cost at loaded parameters: {} \n(this value should be about 22.22)\n'.format(J[0]))

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ============== Part 3: Collaborative Filtering Gradient ==============
# %  Once your cost function matches up with ours, you should now implement
# %  the collaborative filtering gradient function. Specifically, you should
# %  complete the code in cofiCostFunc.m to return the grad argument.
# %
print('\nChecking Gradients (without regularization) ... \n')

# %  Check gradients by running checkNNGradients
checkCostFunction()

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ========= Part 4: Collaborative Filtering Cost Regularization ========
# %  Now, you should implement regularization for the cost function for
# %  collaborative filtering. You can implement it by adding the cost of
# %  regularization to the original cost computation.
# %
# %  Evaluate cost function
J = cofiCostFunc(
    params,
    Y,
    R,
    num_users,
    num_movies,
    num_features,
    1.5
)

print('Cost at loaded parameters (lambda = 1.5): {} \n(this value should be about 31.34)\n'.format(J[0]))

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ======= Part 5: Collaborative Filtering Gradient Regularization ======
# %  Once your cost matches up with ours, you should proceed to implement
# %  regularization for the gradient.
# %

print('\nChecking Gradients (with regularization) ... \n')

# %  Check gradients by running checkNNGradients
checkCostFunction(1.5)

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ============== Part 6: Entering ratings for a new user ===============
# %  Before we will train the collaborative filtering model, we will first
# %  add ratings that correspond to a new user that we just observed. This
# %  part of the code will also allow you to put in your own ratings for the
# %  movies in our dataset!

movie_titles = loadMovieList()

# %  Initialize my ratings
my_ratings = np.zeros(1682, dtype=int)

# % Check the file movie_idx.txt for id of each movie in our dataset
# % For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# # % Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# # % We have selected a few movies we liked / did not like and the ratings we
# # % gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
         print('Rated {} for {}\n'.format(my_ratings[i], movie_titles[i]))

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
#
# %% ================== Part 7: Learning Movie Ratings ====================
# %  Now, you will train the collaborative filtering model on a movie rating
# %  dataset of 1682 movies and 943 users
# %

print('\nTraining collaborative filtering...\n')

mat = io.loadmat('ex8_movies.mat')
R, Y = (
    mat['R'],
    mat['Y']
)
# %  Add our own ratings to the data matrix
my_ratings_2d = my_ratings.reshape((my_ratings.size, 1))
Y = np.hstack((my_ratings_2d, Y))
R = np.hstack((my_ratings_2d != 0, R))

# %  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

# %  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# % Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate((X.flatten(), Theta.flatten()))

# % Set Regularization
L = 10
result = minimize(
    fun=cofiCostFunc,
    x0=initial_parameters,
    args=(
        Y,
        R,
        num_users,
        num_movies,
        num_features,
        L
    ),
    method='CG',
    jac=True,
    options={'maxiter': 50}
).x

# % Unfold the returned theta back into U and W
X = result[:(num_movies * num_features)].reshape(num_movies, num_features)
Theta = result[(num_movies * num_features):].reshape(num_users, num_features)

print('Recommender system learning completed.\n')

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ================== Part 8: Recommendation for you ====================
# %  After training the model, you can now make recommendations by computing
# %  the predictions matrix.
# %

p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean.flatten()
sorted_predictions_idx = np.argsort(my_predictions)[::-1]
print('\nTop recommendations for you:\n')
for i in range(num_features):
    print(
        'Predicting rating {} for movie {}\n'.format(
        my_predictions[sorted_predictions_idx[i]],
        movie_titles[sorted_predictions_idx[i]]
    ))

print('\n\nOriginal ratings provided:\n')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
         print('Rated {} for {}\n'.format(my_ratings[i], movie_titles[i]))

