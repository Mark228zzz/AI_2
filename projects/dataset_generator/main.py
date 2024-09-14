from trainer import Trainer

def main():
    # Initialize Trainer
    trainer = Trainer(epochs=2000)

    # Start training process
    trainer.train()

    # Save the trained Generator and Discriminator models
    trainer.save_models('GEN_2000ep', 'DISC_2000ep')

if __name__ == '__main__':
    main()
