from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


class Trainer:
    def __init__(self, epochs=1000, img_size=64, img_channels=1, classes=10, batch_size=1000,
                 learning_rate=0.0002, device='cuda:0', num_workers=12, betas=(0.5, 0.999), z_dim=100, save_img_every=50):
        # Initialize training parameters
        self.epochs = epochs
        self.img_size = img_size
        self.img_channels = img_channels
        self.classes = classes
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = device
        self.num_workers = num_workers
        self.betas = betas
        self.z_dim = z_dim
        self.save_img_every = save_img_every

        # Create models, optimizers, and dataset
        self.create_models()
        self.loss_optim()
        self.create_dataset()

    def create_models(self):
        # Initialize Generator and Discriminator
        self.gen = Generator(self.classes, self.img_channels, self.img_size, self.z_dim, self.device).to(self.device)
        self.disc = Discriminator(self.classes, self.img_channels, self.img_size).to(self.device)

    def loss_optim(self):
        # Loss function and optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.gen.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.disc.parameters(), lr=self.lr, betas=self.betas)

    def create_dataset(self):
        # Data transformations and loader
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        dataset = datasets.ImageFolder(root='mini_mnist_data', transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train(self):
        for epoch in range(self.epochs):
            for real_imgs, labels in self.dataloader:
                batch_size = real_imgs.size(0)

                # Train Discriminator
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)

                noise = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_labels = torch.randint(0, 10, (batch_size,), device=self.device)
                fake_imgs = self.gen(noise, fake_labels)

                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels_d = torch.zeros(batch_size, 1, device=self.device)

                # Calculate losses for real and fake images
                real_loss = self.criterion(self.disc(real_imgs, labels), real_labels)
                fake_loss = self.criterion(self.disc(fake_imgs.detach(), fake_labels), fake_labels_d)
                loss_D = real_loss + fake_loss

                # Backpropagate and optimize Discriminator
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                noise = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_labels = torch.randint(0, 10, (batch_size,), device=self.device)
                gen_imgs = self.gen(noise, fake_labels)

                # Generator loss (wants Discriminator to classify fake images as real)
                loss_G = self.criterion(self.disc(gen_imgs, fake_labels), real_labels)

                # Backpropagate and optimize Generator
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

            print(f'Epoch: [{epoch+1}/{self.epochs}], loss_G: {loss_G.item()}, loss_D: {loss_D.item()}')

            # Save generated images every few epochs
            if epoch % self.save_img_every == 0:
                j = random.randint(0, 9)
                generated_images = self.gen.generate_images(label=j, num_images=8)
                self.save_imgs(generated_images, f'{j}_ep{epoch}')

    def save_models(self, generator_name, discriminator_name):
        # Save models to file
        torch.save(self.gen.state_dict(), f'models/{generator_name}.pth')
        torch.save(self.disc.state_dict(), f'models/{discriminator_name}.pth')

    def save_imgs(self, generated_imgs, name):
        # Save generated images
        images = generated_imgs.detach().cpu().numpy()
        fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
        for img, ax in zip(images, axes):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
        plt.savefig(f'generated_images/{name}')
        plt.close()
