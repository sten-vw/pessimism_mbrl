import math
import random
import numpy as np
import copy

class Node:
    def __init__(self, action, state, reward, continue_probability, num_actions=3, parent=None, env=None, state_action_map=None, threshold=2):
        # after state
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.values = 0
        self.visits = 0
        self.untried_actions = [i for i in range(num_actions)]
        self.continue_probability = continue_probability
        self.reward = reward
        self.env = env


        for i in self.untried_actions:
            if self.env is not None:
                state_reconstructed = self.env.state()
                state_action = (state_reconstructed.tobytes(), i)
                count = state_action_map.get(state_action, 0)
                if count < threshold:
                    self.untried_actions.remove(i)


    def UCB1(self, total_visits, exploration_constant):
        assert self.visits != 0
        return self.values / self.visits + exploration_constant * math.sqrt(math.log(total_visits) / self.visits)

    def select_child(self, exploration_constant):
        return max(self.children, key=lambda node: node.UCB1(self.visits, exploration_constant))

    def add_child(self, action, child_node):
        self.untried_actions.remove(action)
        self.children.append(child_node)

    def update(self, value):
        self.visits += 1
        self.values += value

    def summarize(self):
        return f"action: {self.action}, value: {round(self.values / self.visits, 2)}, visits: {self.visits}, reward: {round(self.reward, 2)}, continue_probability: {round(self.continue_probability, 2)}"
    
    def __str__(self):
        message = "root node: " + self.summarize() + "\n"
        for child in self.children:
            message += child.summarize() + "\n"
        return message

def MCTS(root: Node, 
         model,
         num_simulations=1000,
         exploration_constant=1.0,
         discount_factor=1.0,
         planning_horizon=math.inf,
         state_action_map=None,
         threshold=2):
    num_actions = model.num_actions

    for _ in range(num_simulations):
        node = root

        depth = 0
        # Selection: find a leaf with untried actions
        while len(node.untried_actions) == 0 and len(node.children) != 0 and depth < planning_horizon:
            node = node.select_child(exploration_constant)
            depth += 1
        if depth < planning_horizon:
            # Expansion: expand this node by adding a child (one of the untried actions)
            assert len(node.untried_actions) != 0
            action = random.choice(node.untried_actions)
            # one-step simulation
            state, reward, done_probability = model.step(node.state, action)

            if node.env is None:
                new_env = None
            else:
                new_env = copy.deepcopy(node.env)
                new_env.act(action)




            continue_probability = 1.0 - done_probability
            child_node=Node(
                action=action,
                state=state,
                parent=node,
                continue_probability=continue_probability,
                reward=reward,
                num_actions=num_actions,
                env=new_env,
                state_action_map=state_action_map,
                threshold=threshold
            )
            node.add_child(action, child_node)
            node = child_node
            depth += 1
            # Roll out: do a random simulation from this state
            cumulative_continuation_probability = 1.0
            cumulative_discount_factor = 1.0
            rollout_discounted_return = 0.0

            if new_env is None:
                temp_env = None
            else:
                temp_env = copy.deepcopy(new_env)
            for _ in range(depth, planning_horizon):
                actions = [i for i in range(num_actions)]
                for i in range(num_actions):
                    if temp_env is not None:
                        state_reconstructed = temp_env.state()
                        state_action = (state_reconstructed.tobytes(), i)
                        count = state_action_map.get(state_action, 0)
                        if count < threshold:
                            actions.remove(i)
                if len(actions) == 0:
                    break
                action = random.choice(actions)
                state, reward, p_done = model.step(state, action)
                if temp_env is not None:
                    temp_env.act(action)
                rollout_discounted_return += reward * cumulative_discount_factor * cumulative_continuation_probability
                cumulative_continuation_probability *= (1.0 - p_done)
                cumulative_discount_factor *= discount_factor
                if cumulative_continuation_probability * cumulative_discount_factor < 1e-6:
                    break
        # Backpropagation
        the_discounted_return = rollout_discounted_return
        while node is not None:
            the_discounted_return = node.reward + discount_factor * node.continue_probability * the_discounted_return
            node.update(the_discounted_return)
            node = node.parent
    
    # return the action with the highest visit count
    return max(root.children, key=lambda node: node.visits).action

def calculate_penalty(state, action, model, L, state_action_map):
    state_reconstructed = model.decode(state)
    state_action = (state_reconstructed.tobytes(), action)
    if state_action_map is not None:
        penalty = L / np.sqrt(state_action_map.get(state_action, 1))
    else:
        penalty = 0.0
    return penalty