from trainer import Trainer

def main():
    trainer = Trainer(num_epochs=25, learning_rate=0.0001)

    # Train the model
    trainer.train_model()

    # Evaluate the model
    trainer.evaluate_model()

if __name__ == '__main__':
    main()