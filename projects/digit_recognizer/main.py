from dataseter import MNISTDataset
from cnn import CNN
from trainer import Trainer
from saver import Saver

if __name__ == "__main__":
    # Initialize the datasets and dataloads the model
    dataset = MNISTDataset(batch_size=64)
    model = CNN()

    # Initialize the trainer with the specified device
    # Check if CUDA (GPU) is available and set the device accordingly
    # If not set to device 'cpu'
    trainer = Trainer(model, dataset.train_loader, dataset.test_loader, lr=0.001, num_epochs=5, device='cuda')

    trainer.train() # Train the model
    trainer.evaluate() # Evaluate the model

    # Init saver
    saver = Saver()
    saver.save_model(model, 'digit_recognizer') # We don`t have to write .pth, it adds automatically
