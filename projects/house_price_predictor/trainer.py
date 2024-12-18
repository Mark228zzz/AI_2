import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple
from preprocess_data import preprocess_data
from network import HousePricePredictor
from plots import plot_loss, plot_predicted_vs_actual


class Trainer:
    def __init__(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        num_workers: int = 0,
        ) -> None:

        # Init hyperparams
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.mean_losses = []

        # Create TrainLoader and TestLoader
        tensor_dataset = preprocess_data()
        self.train_loader, self.test_loader = self.create_dataloaders(*tensor_dataset)

        # Create model, criterion and optimizer
        self.model = HousePricePredictor().to(device)
        self.criterion = nn.HuberLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def create_dataloaders(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
        """
        Create Train and Test Loader for the model.
        Args:
            features: Tensor -> Data of features from data as tensor.
            labels: Tensor -> True values (price) from data as tensor.
        Returns:
            Tuple[DataLoader, DataLoader] -> TrainLoader, TestLoader.
        """
        dataset = TensorDataset(features, labels) # Create a TensorDataset

        dataset_size = len(dataset)

        train_size = int(0.8 * dataset_size) # Get size of the train dataset
        test_size = dataset_size - train_size # Get size of the test dataset

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create train and test DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers)

        return train_loader, test_loader

    def train_model(self) -> None:
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Backpropogation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() # Add sum all of losses during one epoch

            mean_loss = epoch_loss/len(self.train_loader) # Calculate mean loss in each epoch

            print(f'Epoch: [{epoch+1}/{self.num_epochs}], Loss: {mean_loss}')

            self.mean_losses.append(mean_loss) # Add callculated mean loss

    def evaluate_model(self) -> None:
        self.model.eval()

        mse, r2, mae = 0.0, 0.0, 0.0
        self.eval_outputs, self.eval_labels = [], []

        with torch.no_grad():
            for features, labels in self.test_loader:
                outputs = self.model(features)

                labels, outputs = labels.cpu().numpy(), outputs.cpu().numpy() # Transform into cpu and numpy array

                self.eval_outputs.append(outputs)
                self.eval_labels.append(labels)

                # Sum of all each metric
                mse += mean_squared_error(labels, outputs)
                r2 += r2_score(labels, outputs)
                mae += mean_absolute_error(labels, outputs)

        # Calculate mean and print metrics
        print(f"Mean Squared Error on the test loader: {mse/len(self.test_loader):.4f}")
        print(f"R2 Score on the test loader: {r2/len(self.test_loader):.4f}")
        print(f"Mean Absolute Error on the test loader: {mae/len(self.test_loader):.4f}")

        # Concatenate all outputs and labels of evaluation together
        self.eval_outputs, self.eval_labels = np.concat(self.eval_outputs), np.concat(self.eval_labels)

    def plot(self) -> None:
        plot_loss(self.mean_losses)

        plot_predicted_vs_actual(self.eval_outputs, self.eval_labels)
