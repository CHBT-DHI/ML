import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TimeSeriesNH4Dataset(Dataset):
    """
    PyTorch Dataset for time series modeling of SBR data.
    
    Args:
        csv_path (str): Path to the CSV file.
        seq_len (int): Number of timesteps per input sequence.
    """
    def __init__(self, csv_path, seq_len=1):
        self.seq_len = seq_len

        # Load CSV
        df = pd.read_csv(csv_path)

        # Drop the time column
        if '.t' in df.columns:
            df = df.drop(columns=['.t'])

        # Extract features and target
        self.features = df.drop(columns=['.SBR_1.NH4']).values.astype(np.float32)
        self.target = df['.SBR_1.NH4'].values.astype(np.float32)

        # Compute number of sequences
        self.n_samples = len(df) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.target[(idx+self.seq_len)-1]  # predict next step
        return torch.from_numpy(x), torch.tensor(y)

# Example usage:
# dataset = TimeSeriesNH4Dataset("data/raw_normalized/normalized.csv", seq_len=1)
# x, y = dataset[0]
# print(x.shape, y)  # x: (seq_len, n_features), y: scalar
