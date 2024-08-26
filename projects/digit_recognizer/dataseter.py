from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, batch_size=64):
        # Define transformations: Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the traning and testing datasets
        self.train_dataset = datasets.MNIST(root='projects/digit_recognizer/data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='projects/digit_recognizer/data', train=False, download=True, transform=transform)

        # Create data loaders for batching
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
