import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001, num_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss() # Loss function for classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train() # Set model to the train mode

            for images, labels in self.train_loader:
                self.optimizer.zero_grad() # Zero gradient
                outputs = self.model(images) # Forward pass
                loss = self.criterion(outputs, labels) # Calculate the loss
                loss.backward() # Backpropagate the loss
                self.optimizer.step() # Update parameters

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.8f}')

    def evaluate(self):
        self.model.eval() # Set model to test mode
        correct = 0
        total = 0

        with torch.no_grad(): # Turn off gradient
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1) # Get predicted

                total += labels.size(0) # Add to total
                correct += (predicted == labels).sum().item() # Add if predicted equals right answer

                print(f'Predicted: {predicted}, Label: {labels}')

        print(f'Test Accuracy: {100 * correct / total:.3f}%')
