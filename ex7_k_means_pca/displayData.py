# %DISPLAYDATA Display 2D data in a nice grid
# %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
# %   stored in X in a nice grid. It returns the figure handle h and the
# %   displayed array if requested.
#
# % Set example_width automatically if not passed in
# if ~exist('example_width', 'var') || isempty(example_width)
# 	example_width = round(sqrt(size(X, 2)));
# end
#
# % Gray Image
# colormap(gray);
#
# % Compute rows, cols
# [m n] = size(X);
# example_height = (n / example_width);
#
# % Compute number of items to display
# display_rows = floor(sqrt(m));
# display_cols = ceil(m / display_rows);
#
# % Between images padding
# pad = 1;
#
# % Setup blank display
# display_array = - ones(pad + display_rows * (example_height + pad), ...
#                        pad + display_cols * (example_width + pad));
#
# % Copy each example into a patch on the display array
# curr_ex = 1;
# for j = 1:display_rows
# 	for i = 1:display_cols
# 		if curr_ex > m,
# 			break;
# 		end
# 		% Copy the patch
#
# 		% Get the max value of the patch
# 		max_val = max(abs(X(curr_ex, :)));
# 		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
# 		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
# 						reshape(X(curr_ex, :), example_height, example_width) / max_val;
# 		curr_ex = curr_ex + 1;
# 	end
# 	if curr_ex > m,
# 		break;
# 	end
# end
#
# % Display Image
# h = imagesc(display_array, [-1 1]);
#
# % Do not show axis
# axis image off
#
# drawnow;
#
# end

import numpy as np
from matplotlib import pyplot as plt


def displayData(X):
    example_width = np.int(np.round(np.sqrt(X.shape[1])))
    plt.gray()
    m, n = X.shape
    example_height = np.int(n / example_width)
    display_rows = np.int(np.floor(np.sqrt(m)))
    display_cols = np.int(np.ceil(m / display_rows))
    pad = 1
    display_array = np.ones((
        pad + display_rows * (example_height + pad),
        pad + display_cols * (example_width + pad)
    ))
    cur_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if cur_ex >= m:
                break
            max_val = np.max(np.abs(X[cur_ex, :]))
            display_array[
                ((j - 1) * example_height + example_height):(j * example_height + example_height),
                ((i - 1) * example_width + example_width):(i * example_width + example_width)
            ] = X[cur_ex, :].reshape(example_height , example_width) / max_val
            cur_ex = cur_ex + 1
        if cur_ex >= m:
            break

    plt.axis('off')
    plt.imshow(display_array.T)
