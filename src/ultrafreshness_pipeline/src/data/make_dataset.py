import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import TimeSeriesDataset
from .load_data import load_catboost_data


def norm_timeseries(timeseries: list[int]):
    result = torch.stack(
        [
            torch.tensor(ts / np.sum(ts), dtype=torch.float32).view(-1, 1)
            for ts in timeseries
        ],
        dim=0,
    )
    return result


def create_lstm_dataset(dataset: dict[str, list[int]]) -> TimeSeriesDataset:
    keys = list(dataset.keys())
    X = [dataset[key] for key in keys]
    X_prepared = norm_timeseries(X)
    eval_ds = TimeSeriesDataset(X_prepared, keys)
    return eval_ds


def create_lstm_loader(
    eval_ds: TimeSeriesDataset,
    batch_size: int = 512,
    jobs: int = 0,
) -> DataLoader:
    eval_dl = DataLoader(eval_ds, batch_size, shuffle=False, num_workers=jobs)
    return eval_dl


def create_catboost_dataset(
    mr_path: str,
    cluster: str,
    lstm_prob: dict[str, float],
) -> pd.core.frame.DataFrame:
    catboost_features = load_catboost_data(mr_path, cluster)
    lstm_feature = pd.DataFrame(lstm_prob)
    catboost_dataset = catboost_features.join(lstm_feature.set_index('key'), on='key')
    return catboost_dataset
