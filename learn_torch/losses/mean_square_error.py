import torch
import torch.nn as nn

"""
Mean Square Error (MSE) Loss, also known as Quadratic Loss or L2 Loss.
It is commonly used for regression tasks.
"""

# Predicted output tensor
output = torch.tensor([0.5, 0.2, 0.3])

# Actual target tensor
actual = torch.tensor([0.5, 0.3, 0.9])

# Define the Mean Square Error loss function
criterion = nn.MSELoss()

# Calculate the MSE loss between the predicted and actual values
loss = criterion(output, actual).item()

# Print the computed loss value
print(f'{loss = }')
