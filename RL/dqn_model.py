import torch
import torch.nn as nn

import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, n_actions)
        )
        
    def forward(self, x):
        return self.fc(x)
