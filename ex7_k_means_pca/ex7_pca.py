# # %% Machine Learning Online Class
# # %  Exercise 7 | Principle Component Analysis and K-Means Clustering
# # %
# # %  Instructions
# # %  ------------
# # %
# # %  This file contains code that helps you get started on the
# # %  exercise. You will need to complete the following functions:
# # %
# # %     pca.m
# # %     projectData.m
# # %     recoverData.m
# # %     computeCentroids.m
# # %     findClosestCentroids.m
# # %     kMeansInitCentroids.m
# # %
# # %  For this exercise, you will not need to change any code in this file,
# # %  or any other files other than those mentioned above.
# # %
# #
# # %% Initialization
# # clear ; close all; clc
#
import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
import colorsys
from mpl_toolkits.mplot3d import Axes3D
from ex1_linear_regression.featureNormalize import featureNormalize
from ex7_k_means_pca.pca import pca
from ex7_k_means_pca.projectData import projectData
from ex7_k_means_pca.recoverData import recoverData
from ex7_k_means_pca.displayData import displayData
from ex7_k_means_pca.drawLine import drawLine
from ex7_k_means_pca.kMeansInitCentroids import kMeansInitCentroids
from ex7_k_means_pca.runkMeans import runkMeans
from ex7_k_means_pca.plotDataPoints import plotDataPoints

# %% ================== Part 1: Load Example Dataset  ===================
# %  We start this exercise by using a small dataset that is easily to
# %  visualize

print('Visualizing example dataset for PCA.\n\n')

# %  The following command loads the dataset. You should now have the
# %  variable X in your environment
mat = io.loadmat('ex7data1.mat')
X = mat['X']

# %  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis((0.5, 6.5, 2, 8))

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% =============== Part 2: Principal Component Analysis ===============
# %  You should now implement PCA, a dimension reduction technique. You
# %  should complete the code in pca.m

print('\nRunning PCA on example dataset.\n\n')

# %  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# %  Run PCA
U, S = pca(X_norm)


# %  Compute mu, the mean of the each feature
#
# %  Draw the eigenvectors centered at mean of data. These lines show the
# %  directions of maximum variations in the dataset.
# hold on;
drawLine(mu, mu + 1.5 * S[0] * U[:, 0])
drawLine(mu, mu + 1.5 * S[1] * U[:, 1])
# Show mean in red
plt.scatter(mu[0], mu[1], c='r')
plt.show()
# hold off;

print('Top eigenvector: \n')
print(' U(:,0) = {} {} \n'.format(U[0, 0], U[1, 0]))
print('\n(you should expect to see -0.707107 -0.707107)\n')

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% =================== Part 3: Dimension Reduction ===================
# %  You should now implement the projection step to map the data onto the
# %  first k eigenvectors. The code will then plot the data in this reduced
# %  dimensional space.  This will show you what the data looks like when
# %  using only the corresponding eigenvectors to reconstruct it.
# %
# %  You should complete the code in projectData.m
# %
print('\nDimension reduction on example dataset.\n\n')

# %  Plot the normalized dataset (returned from pca)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo', label='Original')
plt.axis((-4, 3, -4, 3))

# %  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {}\n'.format(Z[0]))
print('\n(this value should be about 1.481274)\n\n')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: {} {}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('\n(this value should be about  -1.047419 -1.047419)\n\n')

# %  Draw lines connecting the projected points to the original points
# hold on;
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro', label='Projected')
for i in range (X_norm.shape[0]):
     drawLine(X_norm[i, :], X_rec[i, :])
plt.legend()
plt.show()
# end
# hold off
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =============== Part 4: Loading and Visualizing Face Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  The following code will load the dataset into your environment
# %
print('\nLoading face dataset.\n\n')
#
# %  Load Face dataset
mat = io.loadmat('ex7faces.mat')
X = mat['X']

# %  Display the first 100 faces in the dataset
displayData(X[:100, :])
plt.show()

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
# %% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
# %  Run PCA and visualize the eigenvectors which are in this case eigenfaces
# %  We display the first 36 eigenfaces.
# %
print(['\nRunning PCA on face dataset.\n' '(this might take a minute or two ...)\n\n'])

# %  Before running PCA, it is important to first normalize X by subtracting
# %  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

# %  Run PCA
U, S = pca(X_norm)

# %  Visualize the top 36 eigenvectors found
displayData(U[:, :36])

# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% ============= Part 6: Dimension Reduction for Faces =================
# %  Project images to the eigen space using the top k eigenvectors
# %  If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset.\n\n')

K = 10
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z.shape)

# fprintf('\n\nProgram paused. Press enter to continue.\n');
# pause;
#
# %% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
# %  Project images to the eigen space using the top K eigen vectors and
# %  visualize only using those K dimensions
# %  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 10
X_rec  = recoverData(Z, U, K)

# % Display normalized data
plt.subplot(121)
displayData(X_norm[:100, :])
plt.title('Original faces')

# % Display reconstructed data from only k eigenfaces
plt.subplot(122)
displayData(X_rec[:100, ])
plt.title('Recovered faces')
plt.show()
# axis square;
#
# fprintf('Program paused. Press enter to continue.\n');
# pause;
#
#
# %% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
# %  One useful application of PCA is to use it to visualize high-dimensional
# %  data. In the last K-Means exercise you ran K-Means on 3-dimensional
# %  pixel colors of an image. We first visualize this output in 3D, and then
# %  apply PCA to obtain a visualization in 2D.
#
# close all; close all; clc
#
# % Reload the image from the previous exercise and run K-Means on it
# % For this to work, you need to complete the K-Means assignment first
A = plt.imread('bird_small.png')
#
# % If imread does not work for you, you can try instead
# %   load ('bird_small.mat');
#
A = A / 255
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

# %  Sample 1000 random indexes (since working with all the data is
# %  too expensive. If you have a fast computer, you may increase this.
sel = np.random.permutation(1000)

# %  Setup Color Palette
hsv_tuples = [(x * 1.0 / K, 0.5, 0.5) for x in range(K)]
rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
colors = [rgb_tuples[index - 1] for index in idx[sel, :].flatten()]

# %  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(
    X[sel, 0],
    X[sel, 1],
    X[sel, 2],
    s=10,
    c=colors
)
ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()
# fprintf('Program paused. Press enter to continue.\n')
# pause;
#
# %% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# % Use PCA to project this cloud to 2D for visualization
#
# % Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# % PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# % Plot in 2D
# figure;
plotDataPoints(Z[sel, :], idx[sel, :], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
# fprintf('Program paused. Press enter to continue.\n');
# pause;
