# rl/dqn_model.py

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple Deep Q-Network.
    Input: state vector
    Output: Q-values for each action
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
