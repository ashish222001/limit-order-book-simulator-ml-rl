# rl/train_dqn.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.lob_env import LOBEnv
from rl.dqn_model import DQN
from rl.replay_buffer import ReplayBuffer


# ---------------- HYPERPARAMETERS ----------------
NUM_EPISODES = 50
MAX_STEPS = 100

BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

TARGET_UPDATE_FREQ = 5


# ---------------- TRAINING ----------------
def train():
    env = LOBEnv(max_steps=MAX_STEPS)

    state_dim = len(env.reset())
    action_dim = 3

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(capacity=50_000)

    epsilon = EPS_START
    loss_fn = nn.MSELoss()

    # 🔹 Store episode rewards for plotting
    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            # --- ε-greedy action ---
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_vals = policy_net(s)
                    action = int(torch.argmax(q_vals).item())

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # --- Learn ---
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q = rewards + GAMMA * next_q_values * (1 - dones)

                loss = loss_fn(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 🔹 Log reward
        episode_rewards.append(total_reward)

        print(
            f"Episode {episode:03d} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    # 🔹 Save rewards for plotting
    np.save("rl/dqn_rewards.npy", np.array(episode_rewards))
    print("\nSaved rewards to rl/dqn_rewards.npy")


if __name__ == "__main__":
    train()
