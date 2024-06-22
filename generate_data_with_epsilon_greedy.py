import argparse
import torch
import numpy as np
from collections import namedtuple
from dqn import get_state, QNetwork
from minatar import Environment
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def main():
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--load_file', "-l", type=str)
    parser.add_argument('--output_folder', "-o", type=str, default="data/model_learning_data/test")
    parser.add_argument('--game', "-g", type=str, default='quick_breakout')
    parser.add_argument('--num_frames', "-n", type=int, default=50)
    parser.add_argument('--seed', "-s", type=int, default=42)
    parser.add_argument('--epsilon', "-e", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    env = Environment(args.game, sticky_action_prob=0.0, use_minimal_action_set=True)
    epsilon = args.epsilon

    # seeding
    seed = args.seed
    checkpoint = torch.load(args.load_file, map_location=device)
    random.seed(seed)
    np_seed = random.randint(0, 2**32 - 1)
    env_seed = random.randint(0, 2**32 - 1)
    print("seeds (np, env):", np_seed, env_seed)
    np.random.seed(np_seed)
    env.seed(env_seed)

    num_actions = env.num_actions()
    in_channels = env.state_shape()[2]
    policy_net = QNetwork(in_channels, num_actions).to(device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    # S
    state_trajectories = torch.zeros((args.num_frames, 10, 10, 4), dtype=torch.bool, device=device)
    action_trajectories = torch.zeros((args.num_frames, 1), dtype=torch.int32, device=device)
    reward_trajectories = torch.zeros((args.num_frames, 1), dtype=torch.float32, device=device)
    terminal_trajectories = torch.zeros((args.num_frames, 1), dtype=torch.bool, )

    t = 0
    while t < args.num_frames:

        episode_rewards = 0
        env.seed(0)
        env.reset()
        done = False

        while not done and t < args.num_frames:
            state = get_state(env.state())
            state_trajectories[t] = torch.from_numpy(env.state())
            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(num_actions)]], device=device)
            else:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            reward, done = env.act(action)
            action_trajectories[t] = action
            reward_trajectories[t] = torch.tensor([[reward]], device=device).float()
            terminal_trajectories[t] = torch.tensor([[done]], device=device).float()
            episode_rewards += reward
            t += 1

        # print(f"Episode {i+1} reward: {episode_rewards}")

    os.makedirs(args.output_folder, exist_ok=True)
    np.savez_compressed(f"{args.output_folder}/epsilon_{args.epsilon}_seed_{args.seed}_num_frames_{args.num_frames}.npz",
                        states=state_trajectories[:t].cpu().numpy(),
                        actions=action_trajectories[:t].cpu().numpy(),
                        rewards=reward_trajectories[:t].cpu().numpy(),
                        terminals=terminal_trajectories[:t].cpu().numpy())

if __name__ == '__main__':
    main()