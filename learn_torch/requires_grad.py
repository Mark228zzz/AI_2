import torch

# Creating a tensor with requires_grad=True
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Performing an operation on the tensor
y = x + 2

# Another operation
z = y * y * 3

# Compute the mean
out = z.mean()

# Backpropagate to compute gradients
out.backward()

# Print the gradients
print(x.grad)  # Outputs: tensor([ 6., 12., 18.])


with torch.no_grad():
    y = x + 2 # No gradient will tracked here
