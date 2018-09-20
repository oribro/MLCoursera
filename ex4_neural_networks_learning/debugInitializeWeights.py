# %DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
# %incoming connections and fan_out outgoing connections using a fixed
# %strategy, this will help you later in debugging
# %   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
# %   of a layer with fan_in incoming connections and fan_out outgoing
# %   connections using a fix set of values
# %
# %   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
# %   the first row of W handles the "bias" terms
# %
#
# % Set W to zeros
# W = zeros(fan_out, 1 + fan_in);
#
# % Initialize W using "sin", this ensures that W is always of the same
# % values and will be useful for debugging
# W = reshape(sin(1:numel(W)), size(W)) / 10;
#
# % =========================================================================
#
# end

import numpy as np


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    return np.sin(np.arange(W.size)).reshape(W.shape) / 10

