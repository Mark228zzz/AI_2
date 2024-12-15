import torch.nn as nn


class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()

        # Define the layers of the model
        self.model = nn.Sequential(
            nn.Linear(16, 512),  # Input layer: 16 features -> 512 neurons
            nn.ReLU(),           # ReLU activation
            nn.Dropout(0.2),     # Dropout with 20% chance of turning off

            nn.Linear(512, 256),  # Second hidden layer: 512 -> 256 neurons
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout with 20% chance of turning off

            nn.Linear(256, 128),  # Third hidden layer: 256 -> 128 neurons
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout with 20% chance of turning off

            nn.Linear(128, 1)     # Output layer: 128 -> 1 (price prediction)
        )

    def forward(self, x):
        # Forward pass: the input data flows through the layers
        return self.model(x)