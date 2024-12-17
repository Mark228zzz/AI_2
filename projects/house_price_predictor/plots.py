import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_loss(loss: List[float]) -> None:
    plt.style.use('dark_background')

    plt.figure(figsize=(10, 8))

    plt.plot(loss, label='Loss', color='red', alpha=0.8, linewidth=0.8)
    plt.plot(loss, color='red', alpha=0.2, linewidth=2)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss During the Training', fontsize=16)

    plt.grid(True, alpha=0.2)

    plt.legend()
    plt.show()

def plot_predicted_vs_actual(predicted, actual) -> None:
    plt.figure(figsize=(10, 8))

    # Scatter plot for predicted vs actual values
    plt.scatter(predicted, actual, color='blue', alpha=0.8, label="Predicted", linewidth=0.05)

    # Ideal line where predicted equals actual
    plt.plot([min(predicted), max(predicted)], [min(predicted), max(predicted)], color='green', linestyle='--', label='Ideal', linewidth=1.5)

    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title('Difference between Predicted and Actual', fontsize=16)

    plt.legend()

    plt.show()
