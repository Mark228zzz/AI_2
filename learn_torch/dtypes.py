import torch

# Creating tensors with different data types
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
bool_tensor2 = torch.tensor([True, False, True], dtype=torch.int8)

# Printing the tensors and their data types
print("Float Tensor:", float_tensor)
print("Float Tensor dtype:", float_tensor.dtype, '\n')

print("Double Tensor:", double_tensor)
print("Double Tensor dtype:", double_tensor.dtype, '\n')

print("Int Tensor:", int_tensor)
print("Int Tensor dtype:", int_tensor.dtype, '\n')

print("Long Tensor:", long_tensor)
print("Long Tensor dtype:", long_tensor.dtype, '\n')

print("Bool Tensor:", bool_tensor)
print("Bool Tensor dtype:", bool_tensor.dtype, '\n')

print("Bool Tensor:", bool_tensor2)
print("Bool Tensor dtype:", bool_tensor2.dtype, '\n')

# Changing the data type of a tensor
int_to_float_tensor = int_tensor.to(torch.float32)
print("Converted Int to Float Tensor:", int_to_float_tensor)
print("Converted Tensor dtype:", int_to_float_tensor.dtype)
