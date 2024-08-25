import torch
import torch.nn as nn

"""
Dropout is a regularization technique used to prevent overfitting in neural networks.
It works by randomly "dropping out" (i.e., setting to zero) a fraction of the neurons
during the training phase at each iteration.
"""

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32) # Tensor of data

dropout_50 = nn.Dropout(0.5) # Dropout with 50% probability
dropout_15 = nn.Dropout(0.15) # Dropout with 15% probability

print(f'Without Dropout:     {x}')
print(f'With Dropout 50%:    {dropout_50(x)}')
print(f'With Dropout 15%:    {dropout_15(x)}')

y = torch.tensor([
    [1, 6, 3],
    [6, 3, 4],
    [9, 1, 7]
], dtype=torch.float32)

dropout2d_30 = nn.Dropout2d(0.3)
dropout2d_70 = nn.Dropout2d(0.7)

print(f'Without Dropout:\n{y}')
print(f'With Dropout2d 30%:\n{dropout2d_30(y)}')
print(f'With Dropout2d 70%:\n{dropout2d_70(y)}')
