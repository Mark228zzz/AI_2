import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, classes=10, channels=1, img_size=64, latent_dim=100, device='cuda:0'):
        super(Generator, self).__init__()

        # Initialize parameters
        self.classes = classes  # Number of digit classes (0-9)
        self.channels = channels  # Image channels (e.g., 1 for grayscale)
        self.img_size = img_size  # Size of the generated image
        self.latent_dim = latent_dim  # Latent space size (noise vector)
        self.device = device  # Device for computation (CPU/GPU)
        self.img_shape = (self.channels, self.img_size, self.img_size)

        # Embedding for conditioning the model on class labels
        self.label_embedding = nn.Embedding(self.classes, self.classes)

        # Define the feedforward network layers
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()  # Output scaled between -1 and 1 (for image pixel values)
        )

    def forward(self, noise, labels):
        # Combine noise and label embeddings as input
        z = torch.cat((self.label_embedding(labels), noise), -1)

        # Pass through the model to generate an image
        img = self.model(z)

        # Reshape the output into image dimensions
        return img.view(img.size(0), *self.img_shape)

    def generate_images(self, label, num_images):
        # Generate random noise
        noise = torch.randn(num_images, self.latent_dim, device=self.device)

        # Create a tensor filled with the same label for all images
        labels = torch.full((num_images,), label, dtype=torch.long, device=self.device)

        # Generate and return the images
        return self(noise, labels)
