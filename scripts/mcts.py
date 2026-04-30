import torch
import torch.nn.functional as F
import math
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0

def select_child(node):
    best_score = -float("inf")
    best_action = None
    best_child = None

    for action, child in node.children.items():
        q = child.value_sum / (child.visits + 1e-8)
        u = child.prior * math.sqrt(node.visits) / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

def step_env(env, state, action):
    real_board = env.unwrapped.board.copy()
    real_score = env.unwrapped.total_score
    real_step_score = env.unwrapped.step_score
    real_is_legal = env.unwrapped.is_legal
    real_illegal_count = env.unwrapped.illegal_count

    env.unwrapped.board = state.copy()
    env.unwrapped.total_score = 0
    env.unwrapped.step_score = 0
    env.unwrapped.is_legal = True
    env.unwrapped.illegal_count = 0

    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_board = env.unwrapped.board.copy()
    done = terminated or truncated

    env.unwrapped.board = real_board
    env.unwrapped.total_score = real_score
    env.unwrapped.step_score = real_step_score
    env.unwrapped.is_legal = real_is_legal
    env.unwrapped.illegal_count = real_illegal_count

    return next_board, reward, done

def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent

def expand(node, model):
    state_t = torch.tensor(np.log2(node.state + 1).flatten(), dtype=torch.float32).unsqueeze(0)
    value, logits = model(state_t)
    probs = F.softmax(logits, dim=-1).detach().numpy()

    for action, prob in enumerate(probs):
        child = Node(state=None, parent=node)
        child.prior = prob
        node.children[action] = child

    return value.item()

def run_mcts(env, state, model, num_simulations=20):
    root = Node(state)
    expand(root, model)

    for i in range(num_simulations):
        node = root
        sim_state = state.copy()
        depth = 0
        done = False

        while node.children and depth < 5:
            action, node = select_child(node)
            sim_state, reward, done = step_env(env, sim_state, action)
            node.state = sim_state
            depth += 1
            if done:
                break

        if node.state is not None and not done:
            value = expand(node, model)
        else:
            value = 0.0

        backpropagate(node, value)

    visits = [root.children[a].visits for a in range(4)]
    visits = torch.tensor(visits, dtype=torch.float32)
    policy = (visits / visits.sum()).numpy()

    return policy
