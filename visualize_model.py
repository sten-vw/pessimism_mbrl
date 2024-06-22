import tkinter as tk
import numpy as np
from minatar import Environment
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import copy
import torch
from nets import RepresentationNetwork, DynamicsNetwork, ReconstructionNetwork

import sys
model_folder = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the nets
# model_folder = "data/model_learning/exp5_experience_100_epochs_unroll_5_steps_larger/epoch_70/"
representation_net = torch.load(f"{model_folder}/representation_net.pt", map_location=device)
dynamics_net = torch.load(f"{model_folder}/dynamics_net.pt", map_location=device)
reconstruction_net = torch.load(f"{model_folder}/reconstruction_net.pt", map_location=device)

# Initialize the environment
env = Environment("quick_breakout")
env.seed(0)
state = env.reset()
envs = []
envs.append(copy.deepcopy(env))
zs = []
z = representation_net(torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2))
zs.append(z)
reconstructions = []
reconstruction = reconstruction_net(z).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
reconstructions.append(reconstruction)

# Create a mapping from key symbols to environment actions
key_to_action = {
    'Up': 0, # Assuming 0 is the action for "do nothing"
    'Left': 1,  # Assuming 1 is the action for "move left"
    'Right': 2,  # Assuming 2 is the action for "move right",
    'Down': -1,  # go back,
    'Escape': 404,  # Assuming 404 is the action for "quit",
    'space': 999,  # encode,
    "q": 404
}

# Function to update the environment and visualization based on the key press
def on_key_press(event):
    global envs
    global state
    global env
    global zs
    global z
    global reconstructions
    global reconstruction
    action = key_to_action.get(event.keysym, None)
    print(action)
    if action == 404:
        root.quit()
    elif action == 999:
        z = representation_net(torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2))
        zs[-1] = z
        reconstruction = reconstruction_net(z).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        reconstructions[-1] = reconstruction
        update_visualization(env.state(), reconstruction)
    elif action == -1:
        envs.pop()
        zs.pop()
        reconstructions.pop()
        env = envs[-1]
        z = zs[-1]
        reconstruction = reconstructions[-1]
        state = env.state()
        update_visualization(state, reconstruction)
    elif action is not None:
        _, done = env.act(action)
        # print("encoded state:", z)
        dynamics_net_input = torch.cat((z, torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1) / env.num_actions()), dim=1)
        dynamics_embedding = dynamics_net.embedding_net(dynamics_net_input)
        reward = dynamics_net.reward_net(dynamics_embedding)
        done_logits = dynamics_net.done_net(dynamics_embedding)
        predicted_done = torch.sigmoid(done_logits)
        print("predicted reward", reward.item())
        print("predicted done", predicted_done.item())
        z = dynamics_net.dynamics_net(dynamics_net_input)
        # print("predicted next state:", z)
        if not done:
            state = env.state()
            envs.append(copy.deepcopy(env))
            zs.append(z)
            reconstruction = reconstruction_net(z).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
            reconstructions.append(reconstruction)
            update_visualization(state, reconstruction)
        else:
            env.seed(0)
            env.reset()
            envs.append(copy.deepcopy(env))
            z = representation_net(torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2))
            zs.append(z)
            reconstruction = reconstruction_net(z).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
            reconstructions.append(reconstruction)
            update_visualization(env.state(), reconstruction)

# Function to create and update the environment visualization
def update_visualization(state, reconstructed_state):
    # loss = torch.nn.MSELoss()(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2), reconstruction_net(representation_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)))).item()
    # print(loss)
    # # print(torch.nn.MSELoss()(state, reconstructed_state))
    ax[0].clear()
    ax[1].clear()
    n_channels = 4
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0,0,0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels+2)]
    norm = colors.BoundaryNorm(bounds, n_channels+1)
    numerical_state = np.amax(state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2) + 0.5
    reconstructed_numerical_state = np.amax(reconstructed_state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2) + 0.5
    ax[0].imshow(numerical_state, cmap=cmap, norm=norm, interpolation='none')
    ax[0].axis('off')
    ax[1].imshow(reconstructed_numerical_state, cmap=cmap, norm=norm, interpolation='none')
    ax[1].axis('off')
    canvas.draw()

# Setup Tkinter
root = tk.Tk()
root.title("RL Environment Visualization")
root.bind('<KeyPress>', on_key_press)

# Setup Matplotlib Figure and Tkinter Canvas
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()
plot_widget.pack()

update_visualization(env.state(), reconstructions[-1])  # Initial visualization

root.mainloop()