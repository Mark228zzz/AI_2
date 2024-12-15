import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def get_tensor_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess data using MinMaxScaler and return features and labels as tensors.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Normalized features and labels.
    """
    # Load dataset
    dataset = pd.read_csv('./data/house_data.csv')

    # Convert categorical columns to integers
    for column in ['street', 'city', 'statezip', 'country']:
        dataset[column] = pd.factorize(dataset[column])[0]

    # Separate features (X) and labels (y)
    x = dataset.drop(columns=['price']).values  # Drop target column
    y = dataset['price'].values.reshape(-1, 1)

    # Normalize features and labels using MinMaxScaler
    x_scaler = MinMaxScaler(feature_range=(0, 10))
    y_scaler = MinMaxScaler(feature_range=(0, 10))

    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor
