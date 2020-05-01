"""
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class deepQ(nn.Module):

    def __init__(self, action_size, obs_size, hidden_size=256):
        super().__init__()
        self.input_layer = nn.Linear(obs_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x

