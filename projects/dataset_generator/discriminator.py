import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size):
        super(Discriminator, self).__init__()

        # Initialize parameters
        self.classes = classes  # Number of digit classes
        self.channels = channels  # Image channels (e.g., 1 for grayscale)
        self.img_size = img_size  # Image size
        self.img_shape = (self.channels, self.img_size, self.img_size)

        # Embedding for labels
        self.label_embedding = nn.Embedding(self.classes, self.classes)

        # Discriminator model
        self.model = nn.Sequential(
            nn.Linear(self.classes + int(np.prod(self.img_shape)), 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Outputs a probability of image validity
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)

        # Concatenate image and label
        d_in = torch.cat((img_flat, self.label_embedding(labels)), -1)

        # Predict validity
        validity = self.model(d_in)
        return validity
