import torch
import torch.nn as nn
import torch.nn.functional as F


# Create structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        fc1 = nn.Linear(3, 5)
        fc2 = nn.Linear(5, 2)

    # Create forward pass logic
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)

        return x
