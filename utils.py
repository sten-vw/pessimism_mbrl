import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# visualize the state
def render_state(state):
    n_channels = 4
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0,0,0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels+2)]
    norm = colors.BoundaryNorm(bounds, n_channels+1)
    _, ax = plt.subplots(1,1)
    numerical_state = np.amax(state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2) + 0.5
    ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation='none')
    plt.show()

# visualize the state and its reconstruction
def render_state_reconstruction_window(states, reconstructions, window_size):
    n_channels = 4
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0,0,0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels+2)]
    norm = colors.BoundaryNorm(bounds, n_channels+1)
    numerical_states = np.amax(states * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 3) + 0.5
    numerical_reconstructions = np.amax(reconstructions * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 3) + 0.5
    _, ax = plt.subplots(2, window_size)
    for i in range(window_size):
        ax[0, i].imshow(
            numerical_states[i], cmap=cmap, norm=norm, interpolation='none')
        ax[1, i].imshow(
            numerical_reconstructions[i], cmap=cmap, norm=norm, interpolation='none')
    plt.show()