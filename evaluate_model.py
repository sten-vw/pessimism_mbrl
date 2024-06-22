import torch
from minatar import Environment
import random
import numpy as np
import copy
import fire
import os
import math
from mcts import Node, MCTS
import multiprocessing as multiprocessing
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedModel:

    def __init__(self, num_actions, model):
        self.num_actions = num_actions
        representation_net, dynamics_net = model
        self.dynamics_net = dynamics_net
        self.representation_net = representation_net

    def step(self, z, action):
        # encode the action as one hot encoding

        # action = F.one_hot(torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).long(), num_classes=self.num_actions)
        action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1) / self.num_actions
        net_input = torch.cat([z, action], dim=1)
        with torch.no_grad():
            next_z, reward, done = self.dynamics_net(net_input)
        return next_z, reward.item(), done.item()
    
    def encode(self, env):
        z = self.representation_net(torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2))
        return z

class TrueModel:

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, env, action):
        copied_env = copy.deepcopy(env)
        reward, terminal = copied_env.act(action)
        return copied_env, reward, terminal
    
    def encode(self, env):
        return copy.deepcopy(env)
    
def evaluate_episode(param):

        model_folder, eval_seed, mcts_params = param

        # load the models
        representation_net = torch.load(f'{model_folder}/representation_net.pt', map_location=device)
        dynamics_net = torch.load(f'{model_folder}/dynamics_net.pt', map_location=device)
        reconstruction_net = torch.load(f'{model_folder}/reconstruction_net.pt', map_location=device)

        # set the models to evaluation mode
        representation_net.eval()
        dynamics_net.eval()
        reconstruction_net.eval()

        # make the environment and model
        env = Environment('quick_breakout', sticky_action_prob=0.0, use_minimal_action_set=True)
        learned_model = LearnedModel(env.num_actions(), (representation_net, dynamics_net))
        model = learned_model

        # set the random seeds
        random.seed(eval_seed)
        env_seed = random.randint(0, 2**31)
        torch_seed = random.randint(0, 2**31)
        np_seed = random.randint(0, 2**31)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed(torch_seed)
        np.random.seed(np_seed)
        env.seed(env_seed)

        # reset the environment
        env.reset()

        # get mcts hyperparameters from mcts_params
        exploration_constant = mcts_params["exploration_constant"]
        discount_factor = mcts_params["discount_factor"]
        num_simulations = mcts_params["num_simulations"]
        planning_horizon = mcts_params["planning_horizon"] if "planning_horizon" in mcts_params else math.inf

        # the agent-environment loop
        with torch.inference_mode():
            undiscounted_return = 0
            done = False
            t = 0
            while not done:
                z = model.encode(env)
                root = Node(
                    state=z,
                    reward=0.0,
                    continue_probability=1.0,
                    parent=None,
                    action=None,
                    num_actions=env.num_actions()
                )
                best_action = MCTS(root, model, num_simulations, exploration_constant, discount_factor, planning_horizon)
                reward, done = env.act(best_action)
                undiscounted_return += reward
                t += 1
                if t >= 1000:
                    break
        print(t, undiscounted_return)
        return undiscounted_return

def main(output_folder: str, model_folder: str, seed: int, num_episodes: int, exploration_constant: float = 1.0, discount_factor: float = 0.97, planning_horizon: int = 32, num_simulations: int = 128):

    os.makedirs(output_folder, exist_ok=True)
    
    # save params
    with open(f"{output_folder}/params.txt", 'w') as f:
        f.write(f"model_folder: {model_folder}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"exploration_constant: {exploration_constant}\n")
        f.write(f"discount_factor: {discount_factor}\n")
    
    # seeding
    random.seed(seed)
    print("master seed:", seed)
    # generate random seeds for each episode
    eval_seeds = [random.randint(0, 2**31) for _ in range(num_episodes)]

    mcts_params = {
        "exploration_constant": exploration_constant,
        "discount_factor": discount_factor,
        "num_simulations": num_simulations,
        "planning_horizon": planning_horizon if planning_horizon is not None else math.inf
    }

    job_params = [(model_folder, eval_seeds[i], mcts_params) for i in range(num_episodes)]
    print(f"jobs created with seeds {eval_seeds}.")

    cpu_count = multiprocessing.cpu_count()
    num_processes = int(cpu_count / 4)
    with multiprocessing.get_context('spawn').Pool(num_processes) as pool:
        episode_returns = pool.map(evaluate_episode, job_params)

    # save results
    np.save(f"{output_folder}/episode_returns.npy", np.array(episode_returns))
    print(f"Episode returns: {episode_returns}")
    print(f"Mean episode return: {np.mean(episode_returns)}")

if __name__ == '__main__':
    fire.Fire(main)