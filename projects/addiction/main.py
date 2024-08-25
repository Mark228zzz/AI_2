from dqn_trainer import Trainer

def main():
    trainer = Trainer(batch_size=256, max_steps_per_episode=50)
    trainer.learn()

if __name__ == '__main__':
    main()
