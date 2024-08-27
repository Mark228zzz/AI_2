from trainer import Trainer

def main():
    trainer = Trainer(device='cuda', map_size=10, max_steps=40, cell_size=100)

    trainer.train()

if __name__ == '__main__':
    main()
