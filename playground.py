import numpy as np


trajectory_data = np.load('data/experience_data_episode_10000.npz')
states, actions, rewards, terminals = trajectory_data['states'], trajectory_data['actions'], trajectory_data['rewards'], trajectory_data['terminals']
#Create dictionary of counts for each state-action pair
state_action_counts = {}
for state, action in zip(states, actions):
    state_action = (state.tobytes(), action.tobytes())
    if state_action in state_action_counts:
        state_action_counts[state_action] += 1
    else:
        state_action_counts[state_action] = 1
print(state_action_counts)