# Dataset Generator Using Conditional GAN (cGAN)

This project demonstrates how to build a dataset generator using Conditional GAN (cGAN) with PyTorch. We utilize the MNIST dataset to train a generator capable of producing images of handwritten digits based on specified labels.
Getting Started.

**Step 1: Extract Data:**

To begin, run the extract_data.py script. This will automatically install the MNIST dataset and create a folder named mini_mnist_data containing 100 images for each digit (0-9).

```bash
python extract_data.py
```

**Step 2: Train the Model:**

Next, run the main.py file to start the training process:

```bash
python main.py
```

During training, two models will be created:

    DISCRIMINATOR – Used only during training to help improve the generator.
    GENERATOR – The main model that will be used to generate images after training.

You can adjust the training parameters such as the number of epochs, batch size, and learning rate in the main.py file to experiment with different hyperparameters for better performance.

**Step 3: Generate Images:**

Once training is complete, the GENERATOR model can be used to generate images on demand. It produces tensors in the format `[num_imgs, 1, 64, 64]`. For example, to generate 20 images of the digit "4", the generator will return a tensor of shape `[20, 1, 64, 64]`.

**Step 4: Test the Model:**

To test the trained generator, edit the test_generator.py file by specifying the path to your GENERATOR model (e.g., models/GEN_2000ep.pth). Then, run the script to generate 10 images for each digit (0-9).

```bash
python test_generator.py
```

This script will create and save the generated images in the generated_images folder with filenames corresponding to each digit.

## Summary

    1. Run extract_data.py to install MNIST and create the mini_mnist_data folder.
    2. Run main.py to train the cGAN model. You can adjust the training parameters to suit your needs.
    3. After training, the GENERATOR model can be used to generate custom image datasets.
    4. Use test_generator.py to load the generator model and generate sample images of digits 0-9.Dataset Generator Using Conditional GAN (cGAN)

This project demonstrates how to build a dataset generator using Conditional GAN (cGAN) in PyTorch. It uses the MNIST dataset to train a generator that can produce images of handwritten digits.
Getting Started
Step 1: Extract Data

To begin, run the extract_data.py script. This will automatically install the MNIST dataset and create a folder named mini_mnist_data with 100 images for each digit (0–9).

bash

python extract_data.py

Step 2: Train the Model

Next, run the main.py file to start the training process:

bash

python main.py

During training, two models will be created:

    DISCRIMINATOR – Used only during training.
    GENERATOR – This model is what you will use to generate images after training.

You can modify the training parameters (epochs, batch size, learning rate, etc.) in the main.py file to find better hyperparameters that suit your needs.
Step 3: Generate Images

After training, the GENERATOR model will generate images. It produces tensors in the shape [num_imgs, 1, 64, 64]. For example, to generate 20 images of the digit "4", it will return a tensor shaped [20, 1, 64, 64].
Step 4: Test the Model

To test the trained generator, go to the test_generator.py file. You will need to specify the path to the GENERATOR model (e.g., models/GEN_2000ep.pth) in the script. Then, run the file to generate 10 images of each digit (0-9).

bash

python test_generator.py

Summary

    Run extract_data.py to install MNIST and create a folder mini_mnist_data.
    Run main.py to start training the cGAN. Adjust the training parameters as needed.
    Only the GENERATOR model is required after training to generate images.
    To test the model, modify the path in test_generator.py and run it to generate sample images.