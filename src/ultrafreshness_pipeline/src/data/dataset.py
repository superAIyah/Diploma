import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, keys: list[str]):
        self.X = X
        self.keys = keys

        assert len(self.X) == len(self.keys)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx], self.keys[idx]
