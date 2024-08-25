import torch
import torch.nn.functional as F

x = torch.tensor([1.2, -0.5, 2.1, -1.8, 2.3])

print(f'x:          {x}')
print(f'ReLU:       {F.relu(x)}')
print(f'Leaky ReLU: {F.leaky_relu(x, negative_slope=0.01)}')
print(f'Sigmoid:    {F.sigmoid(x)}')
print(f'Tanh:       {F.tanh(x)}')
print(f'ELU:        {F.elu(x)}')
print(f'Softmax:    {F.softmax(x, dim=0)}')
print(f'Softplus:   {F.softplus(x, beta=0.9)}\n')

y = torch.tensor([
    [1.5, 3.2, -0.2],
    [-1.1, 1.9, 2.2]
])

print(f'Softmax dim=0:\n{F.softmax(y, dim=0)}')
print(f'Softmax dim=1:\n{F.softmax(y, dim=1)}')
