import torch.nn as nn


# Initialize Deep Neural Regression Model
class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()

        # Define the layers of the model
        self.model = nn.Sequential(
            nn.Linear(16, 1024),  # Input layer: 16 features -> 1024 neurons
            nn.ReLU(),           # ReLU activation
            nn.Dropout(0.2),     # Dropout with 20% chance of turning off

            nn.Linear(1024, 512),  # Second hidden layer: 1024 -> 512 neurons
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout with 20% chance of turning off

            nn.Linear(512, 256),  # Third hidden layer: 512 -> 256 neurons
            nn.ReLU(),            # ReLU activation
            nn.Dropout(0.2),      # Dropout with 20% chance of turning off

            nn.Linear(256, 1)     # Output layer: 256 -> 1 (price prediction)
        )

    def forward(self, x):
        # Forward pass: the input data flows through the layers
        return self.model(x)
