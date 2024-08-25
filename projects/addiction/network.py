import torch.nn as nn
import torch.nn.functional as F


# create Network, matrix_of_state-128-128-4
class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Forward pass, ReLU
        x = F.relu(self.fc2(x)) # Forward pass, ReLU
        x = self.fc3(x) # Forward pass Q values

        return x
