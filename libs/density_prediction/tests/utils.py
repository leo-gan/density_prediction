import glob

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

SAMPLE_SIZE = 10000
SAMPLES_IN_FILE = 48
ROWS_IN_FILE = SAMPLE_SIZE * SAMPLES_IN_FILE

INPUT_DIR = "../data/reduced/2000"
INPUT_FILE_PATTERN = "2000-*.parquet"
DATA_FIELDS = ["datetime", "density"]

data_file = "data/reduced/2000/2000-01-01.parquet"


def load_data_from_parquet(parquet_files, verbose=False):
    """load and preprocess data from Parquet files."""
    data_frames = []
    for file in parquet_files:
        df = pd.read_parquet(file, columns=DATA_FIELDS)
        if verbose:
            print(f"  Loaded {file} {df.shape}")
        else:
            print(".", end="")
        data_frames.append(df)
    combined_df = pd.concat(data_frames).reset_index(drop=True)
    return combined_df


def create_sequences(data, seq_length):
    """Create input sequences and their respective targets (shifted by one step)."""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequence = data[i : i + seq_length]
        target = data[i + seq_length]  # The next time step
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def get_dataloader(
    sequence_length=SAMPLES_IN_FILE,
    batch_size=32,
    tensor_dtype=torch.float32,
    verbose=False,
    files=None,
    train_split=0.8,
):
    # Load the data from files
    parquet_files = glob.glob(f"{INPUT_DIR}/{INPUT_FILE_PATTERN}")[:files]
    data = load_data_from_parquet(parquet_files)
    print(f"Loaded {len(parquet_files)} files, {data.shape}")

    # Extracting 'density' values as input samples, and creating shifted targets
    density_values = data["density"].values.reshape(
        -1, SAMPLE_SIZE
    )  # Reshape into (n_samples, SAMPLE_SIZE)
    if verbose:
        print(f"  {density_values.shape=}")

    # Normalize the density values
    scaler = MinMaxScaler()
    density_normalized = scaler.fit_transform(density_values)
    print(f"Normalized {len(density_normalized) = }, {density_normalized[0] = }")

    sequences, targets = create_sequences(density_normalized, sequence_length)

    # Convert to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=tensor_dtype)
    targets = torch.tensor(targets, dtype=tensor_dtype)
    if verbose:
        print(f"  {sequences.shape=}")
        print(f"  {targets.shape=}")

    # Split data into training and validation sets (80% train, 20% validation)
    train_size = int(train_split * len(sequences))
    val_size = len(sequences) - train_size
    train_dataset, val_dataset = random_split(
        TensorDataset(sequences, targets), [train_size, val_size]
    )
    print(f"Created TensorDatasets: {train_dataset=}, {val_dataset=}")

    # Create DataLoaders without shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Created DataLoaders: {train_loader=}, {val_loader=}, {batch_size=}")
    return train_loader, val_loader, density_normalized, scaler
