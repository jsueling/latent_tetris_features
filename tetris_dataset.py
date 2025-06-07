import os

import numpy as np
import torch
from torch.utils.data import Dataset

class TetrisDataset(Dataset):
    """Custom dataset for loading Tetris game states from .npy files."""

    def __init__(self, data_dir="./data", device=None):
        """
        Args:
            data_dir: Directory containing .npy files
            device: Device to move tensors to (cuda/cpu)
        """
        assert device is not None, "Initialise with the device used in the model"

        os.makedirs(data_dir, exist_ok=True)
        self.device = device

        # Find all .npy files in data_dir
        file_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.npy')
        ]

        if not file_paths:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.samples = []
        for file_path in file_paths:
            data = np.load(file_path)
            for sample in data:
                assert sample.shape == (207,), \
                    f"Expected sample shape (207,), got {sample.shape}"
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single game state."""
        return torch.Tensor(self.samples[idx]).to(self.device)
