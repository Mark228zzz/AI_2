import os
from torchvision import datasets, transforms

# Create a directory structure
main_dir = "my_mnist_images"
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

# Create subdirectories for each digit
for i in range(10):
    digit_dir = os.path.join(main_dir, str(i))
    if not os.path.exists(digit_dir):
        os.makedirs(digit_dir)

# Load the MNIST dataset
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Counters for each digit
digit_count = {str(i): 0 for i in range(10)}

# Extract 100 images per digit
for img, label in mnist_train:
    digit = str(label)
    if digit_count[digit] < 100:
        # Convert tensor to PIL image
        img_pil = transforms.ToPILImage()(img)

        # Save the image
        img_save_path = os.path.join(main_dir, digit, f"{digit_count[digit]}.jpg")
        img_pil.save(img_save_path)

        # Increment the count
        digit_count[digit] += 1

    # Stop when 100 images of each digit are saved
    if all(count == 100 for count in digit_count.values()):
        break

print("Extraction complete!")
