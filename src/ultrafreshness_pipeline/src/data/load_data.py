import pandas as pd
import yt.wrapper


def yt_load(mr_path: str, cluster: str) -> yt.wrapper.format.RowsIterator:
    client = yt.wrapper.YtClient(proxy=cluster)
    return client.read_table(mr_path)


def load_lstm_data(mr_path: str, cluster: str) -> dict[str, list[int]]:
    dataset = yt_load(mr_path, cluster)
    dataset_d = {}
    for elem in dataset:
        dataset_d[elem['key']] = elem['timeseries']

    return dataset_d


def load_catboost_data(mr_path: str, cluster: str) -> pd.DataFrame:
    dataset = yt_load(mr_path, cluster)
    return pd.DataFrame(dataset)
