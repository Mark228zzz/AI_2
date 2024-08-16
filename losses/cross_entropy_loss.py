import torch
import torch.nn as nn

"""
Model makes forward pass and the last layer is the logits we have.
Model -> outputs (or logits) -> softmax -> CrossEntropyLoss -> float number
which shows how good did model predict. But we don`t have to
apply softmax to the logits because CrossEntropyLoss automatically
applies softmax to the logits.
"""

# Model output (logits) for 3 classes
outputs = torch.tensor([2.0, 1.0, 0.1, 0.6]).unsqueeze(0)

# Target class (the correct class is the second one, index 1)
target = torch.tensor([0])

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, target).item()
print(f'{loss = }')
