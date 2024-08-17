import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DNN (Deep Neural Network).
DNN refers to a neural network with multiple layers
between input and output layer named hidden layers.
DNN is a type of artificial neural network (ANN).

Our structure is:

3-6-5-2

3 - input neurons (input layer),
6 - first hidden neurons (hidden layer),
5 - second hidden neurons (hidden layer),
2 - output neurons (output layer).

We can put in our Deep Neural Network any number
of hidden layers, but there can be only one input
layer and one output layer.
"""


# Create structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        fc1 = nn.Linear(3, 6)
        fc2 = nn.Linear(6, 5)
        fc3 = nn.Linear(5, 2)

    # Create forward pass logic
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)

        return x
