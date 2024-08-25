import torch

x = torch.tensor([5, 2, 8, 3])

print(f'x:\n{x}')
print(f'{x.shape}\n')

print(f'x.unsqueeze(0):\n{x.unsqueeze(0)}')
print(f'{x.unsqueeze(0).shape}\n')

print(f'x.unsqueeze(1):\n{x.unsqueeze(1)}')
print(f'{x.unsqueeze(1).shape}\n')

print(f'x.unsqueeze(-1):\n{x.unsqueeze(-1)}')
print(f'{x.unsqueeze(-1).shape}\n')

y = torch.tensor([
    [6, 3, 9, 1],
    [2, 5, 7, 8]
])

print(f'y:\n{y}')
print(f'{y.shape}\n')

print(f'y.unsqueeze(0):\n{y.unsqueeze(0)}')
print(f'{y.unsqueeze(0).shape}\n')

print(f'y.unsqueeze(1):\n{y.unsqueeze(1)}')
print(f'{y.unsqueeze(1).shape}\n')

print(f'y.unsqueeze(-1):\n{y.unsqueeze(-1)}')
print(f'{y.unsqueeze(-1).shape}\n')

print(f'y.unsqueeze(2):\n{y.unsqueeze(2)}')
print(f'{y.unsqueeze(2).shape}\n')
