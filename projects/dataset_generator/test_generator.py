from generator import Generator
import torch
import matplotlib.pyplot as plt

device = 'cuda:0'

gen = Generator(device=device).to(device)
gen.load_state_dict(torch.load('models/GEN_2000ep.pth'))

def save_imgs(generated_imgs, name):
    images = generated_imgs.detach().cpu().numpy()
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
    for img, ax in zip(images, axes):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'generated_images/{name}')
    plt.close()

for i in range(10):
    save_imgs(gen.generate_images(label=i, num_images=10), f'{i}')
