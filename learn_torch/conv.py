import torch
import torch.nn as nn


class Conv1dExample(nn.Module):
    def __init__(self):
        super(Conv1dExample, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1d(x)

        return x


class Conv2dExample(nn.Module):
    def __init__(self):
        super(Conv2dExample, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d(x)

        return x


class Conv3dExample(nn.Module):
    def __init__(self):
        super(Conv3dExample, self).__init__()

        self.conv3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3d(x)

        return x


model1d = Conv1dExample()
model2d = Conv2dExample()
model3d = Conv3dExample()

input_data1d = torch.randn(1, 1, 100) # Batch size of 1, 1 channel, length of 100
input_data2d = torch.randn(1, 3, 64, 64) # Batch size of 1, 3 channel, 64x64 image
input_data3d = torch.randn(1, 1, 16, 64, 64) # Batch size of 1, 1 channel, depth 16, 64x64 image

print(model1d(input_data1d))
print(model2d(input_data2d))
print(model3d(input_data3d))
