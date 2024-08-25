import torch
import torch.nn as nn

x = torch.tensor([1.0, -0.5, 3.0], requires_grad=True)
layer = nn.LeakyReLU(0.2, inplace=False)
y = layer(x)

print(x, y)

r = torch.tensor([1.0, -0.5, 3.0])
layer_inplace = nn.LeakyReLU(0.2, inplace=True)
t = layer_inplace(r)

print(r, t)
