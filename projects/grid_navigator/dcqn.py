import torch.nn as nn
import torch.nn.functional as F


# Define the Deep Convolutional Q-Network (DCQN)
class DCQN(nn.Module):
    def __init__(self, state_shape, action_size):
        super(DCQN, self).__init__()

        # Convolutional layers to extract features from the input state
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers to estimate Q-values for each action
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
