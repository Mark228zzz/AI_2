from trainer import Trainer

def main():
    trainer = Trainer(batch_size=64, max_steps_per_episode=200)
    trainer.learn()

if __name__ == '__main__':
    main()
