import torch


class Saver:
    def save_model(self, model, name):
        # Save model to saved_models folder
        torch.save(model.state_dict(), f'projects/digit_recognizer/saved_models/{name}.pth')
