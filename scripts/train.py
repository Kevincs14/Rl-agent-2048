import gymnasium_2048
import gymnasium as gym
import torch
import torch.nn.functional as F
from model import PolicyNetwork
from mcts import run_mcts

env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode="human")
model = PolicyNetwork(state_dim=16, action_dim=4, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

gamma = 0.99
log_file = open("training_log.txt", "w")

for episode in range(500):
    obs, _ = env.reset()
    state = env.unwrapped.board.copy()
    done = False
    total_reward = 0

    while not done:
        state_t = torch.tensor(state.flatten(), dtype=torch.float32)
        state_value, logits = model(state_t)

        mcts_policy = run_mcts(env, state, model)
        action = torch.multinomial(torch.tensor(mcts_policy), 1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = env.unwrapped.board.copy()
        next_state_t = torch.tensor(next_state.flatten(), dtype=torch.float32)

        next_state_value, _ = model(next_state_t)

        td_target = reward + gamma * next_state_value.detach()
        td_error = td_target - state_value

        log_probs = F.log_softmax(logits, dim=-1)
        actor_loss = -torch.sum(torch.tensor(mcts_policy) * log_probs)
        critic_loss = td_error.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        done = terminated or truncated
        total_reward += reward

    board = env.unwrapped.board
    max_tile = int(2 ** board.max()) if board.max() > 0 else 0
    winning_rate = 1.0 if max_tile >= 2048 else 0.0
    mean_score = total_reward

    log_line = (
        f"episode {episode+1}: winning rate = {winning_rate:.2f}, "
        f"mean score = {mean_score:.2f}, max tile = {max_tile}\n"
    )
    print(log_line, end="")
    log_file.write(log_line)
    log_file.flush()

log_file.close()
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved.")