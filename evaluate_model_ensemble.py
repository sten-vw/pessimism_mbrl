import torch
from minatar import Environment
import random
import numpy as np
import copy
import fire
import os
import math
from mcts_ensemble import Node, MCTS
import multiprocessing as multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedModel:

    def __init__(self, num_actions, model):
        self.num_actions = num_actions
        representation_nets, dynamics_nets, reconstruction_nets = model
        self.dynamics_nets = dynamics_nets
        self.representation_nets = representation_nets
        self.reconstruction_nets = reconstruction_nets

    def step(self, states, action):
        action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1) / self.num_actions
        with torch.no_grad():
            net_inputs = [torch.cat([z, action], dim=1) for z in states]
            outputs = [dynamics_net(net_input) for dynamics_net, net_input in zip(self.dynamics_nets, net_inputs)]
            next_latent_states = [output[0] for output in outputs]
            next_rewards = [output[1] for output in outputs]
            next_dones = [output[2] for output in outputs]
            penalty = torch.std(torch.stack(next_rewards), dim=0).mean().item()
            reward = torch.mean(torch.stack(next_rewards), dim=0)
            done = torch.mean(torch.stack(next_dones), dim=0)
        return next_latent_states, reward.item(), done.item(), penalty
    
    def encode(self, env):
        outputs = [representation_net(torch.tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)) for representation_net in self.representation_nets]
        return outputs

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

        model_folder, eval_seed, mcts_params, n_ensembles = param

        representation_nets = []
        dynamics_nets = []
        reconstruction_nets = []


        # load the models
        for i in range(n_ensembles):
            representation_net = torch.load(f'{model_folder}/{i}/representation_net.pt', map_location=device)
            dynamics_net = torch.load(f'{model_folder}/{i}/dynamics_net.pt', map_location=device)
            reconstruction_net = torch.load(f'{model_folder}/{i}/reconstruction_net.pt', map_location=device)
            # set the models to evaluation mode
            representation_net.eval()
            dynamics_net.eval()
            reconstruction_net.eval()

            representation_nets.append(representation_net)
            dynamics_nets.append(dynamics_net)
            reconstruction_nets.append(reconstruction_net)


        # make the environment and model
        env = Environment('quick_breakout', sticky_action_prob=0.0, use_minimal_action_set=True)
        learned_model = LearnedModel(env.num_actions(), (representation_nets, dynamics_nets, reconstruction_nets))
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
        penalty = mcts_params["penalty"] if "penalty" in mcts_params else 0.0

        # the agent-environment loop
        with torch.inference_mode():
            undiscounted_return = 0
            done = False
            t = 0
            while not done:
                zs = model.encode(env)
                root = Node(
                    state=zs,
                    reward=0.0,
                    continue_probability=1.0,
                    parent=None,
                    action=None,
                    num_actions=env.num_actions()
                )
                best_action = MCTS(root, model, num_simulations, exploration_constant, discount_factor,
                                   planning_horizon, penalty)
                reward, done = env.act(best_action)
                undiscounted_return += reward
                t += 1
                if t >= 1000:
                    break
        print(t, undiscounted_return)
        return undiscounted_return

def main(output_folder: str, model_folder: str, penalty: float, seed: int, num_episodes: int, n_ensembles: int = 5, exploration_constant: float = 1.0, discount_factor: float = 0.97, planning_horizon: int = 32, num_simulations: int = 128):

    os.makedirs(output_folder, exist_ok=True)
    
    # seeding
    random.seed(seed)
    print("master seed:", seed)
    # generate random seeds for each episode
    eval_seeds = [random.randint(0, 2**31) for _ in range(num_episodes)]

    mcts_params = {
        "exploration_constant": exploration_constant,
        "discount_factor": discount_factor,
        "num_simulations": num_simulations,
        "planning_horizon": planning_horizon if planning_horizon is not None else math.inf,
        "penalty": penalty
    }

    job_params = [(model_folder, eval_seeds[i], mcts_params, n_ensembles) for i in range(num_episodes)]
    print(f"jobs created with seeds {eval_seeds}.")

    if "SLURM_JOB_CPUS_ON_NODE" in os.environ:
        cpu_count = int(os.environ["SLURM_JOB_CPUS_ON_NODE"])
    elif "SLURM_CPUS_ON_NODE" in os.environ:
        cpu_count = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        cpu_count = multiprocessing.cpu_count()
    print(f"cpu count: {cpu_count}")
    num_processes = int(cpu_count - 1)
    with multiprocessing.get_context('spawn').Pool(num_processes) as pool:
        episode_returns = pool.map(evaluate_episode, job_params)

    # save results
    np.save(f"{output_folder}/episode_returns.npy", np.array(episode_returns))
    print(f"Episode returns: {episode_returns}")
    print(f"Mean episode return: {np.mean(episode_returns)}")

    # save params
    with open(f"{output_folder}/log.txt", 'w') as f:
        f.write(f"model_folder: {model_folder}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"exploration_constant: {exploration_constant}\n")
        f.write(f"discount_factor: {discount_factor}\n")
        f.write(f"penalty: {penalty}\n")
        f.write(f"model_folder: {model_folder}\n")
        f.write(f"Episode returns: {episode_returns}\n")
        f.write(f"Mean episode return: {np.mean(episode_returns)}\n")

if __name__ == '__main__':
    fire.Fire(main)