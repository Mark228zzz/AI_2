import torch
import torch.nn as nn

"""
CrossEntropyLoss or LogLoss.
Model makes forward pass and the last layer is the logits we have.
Model -> outputs (or logits) -> softmax -> CrossEntropyLoss -> float number
which shows how good did model predict. But we don`t have to
apply softmax to the logits because CrossEntropyLoss automatically
applies softmax to the logits.
For example we have 3 classes: {1: dog, 2}. There is predicted values after softmax:
[0.1, 0.6, 0.3]
and actual values:
[0, 1, 0]
It means that our NN thinks that:
10% - class 1
60% - class 2
30% - class 3
And then DNN chooses the biggest one.
"""

# Model output (logits) for 4 classes
outputs = torch.tensor([2.0, 1.0, 0.1, 0.6]).unsqueeze(0)

# Target class (the correct class is the second one, index 1)
target = torch.tensor([1])

# Define the CrossEntropyLoss function
criterion = nn.CrossEntropyLoss()

# Calculate the CrossEntoryLoss between the predicted and actual values
loss = criterion(outputs, target).item()

# Print the loss
print(f'{loss = }')
