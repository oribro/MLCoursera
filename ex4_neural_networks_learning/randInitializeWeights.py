# %RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
# %incoming connections and L_out outgoing connections
# %   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
# %   of a layer with L_in incoming connections and L_out outgoing
# %   connections.
# %
# %   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
# %   the first column of W handles the "bias" terms
# %
#
# % You need to return the following variables correctly
# W = zeros(L_out, 1 + L_in);
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: Initialize W randomly so that we break the symmetry while
# %               training the neural network.
# %
# % Note: The first column of W corresponds to the parameters for the bias unit
# %
#
#
#
#
#
#
#
#
#
# % =========================================================================
#
# end

import numpy as np

def randInitializeWeights(L_in, L_out):
    # Use the suggested formula from ex description
    epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init