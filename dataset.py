import dateutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
#from data import load_signal


class HaiDataset(Dataset):
    def __init__(self, vals, window_size=90):

        self.vals = torch.FloatTensor(vals)
        # self.vals = df.iloc[:, 1:].values
        self.__len = vals.shape[0] - window_size + 1
        self.window_size = window_size

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        return self.vals[idx : idx + self.window_size, :]


DROP_COLS = [
    "C02",
    "C09",
    "C10",
    "C18",
    "C19",
    "C22",
    "C26",
    "C29",
    "C36",
    "C38",
    "C39",
    "C49",
    "C52",
    "C55",
    "C63",
    "C69",
    "C82",
    "C85",
    "C34",
    "C46",
    "C48",
    "C61",
    "C64",
]


def nyc_dataset(window_size=100):
    signal = "nyc_taxi"
    df = load_signal(signal)
    vals = df["value"].to_numpy().reshape(-1, 1)
    vals = MinMaxScaler().fit_transform(vals)
    ds = HaiDataset(vals, window_size)
    return ds


def get_dataset(window_size=100, cols=None, _type="train", dataroot="./data"):
    assert _type in ['train', 'val']

    dataroot = Path(dataroot)

    tr_dfs = [pd.read_csv(fn) for fn in (dataroot / "train").glob("*.csv")]
    va_df = pd.read_csv(dataroot / "validation/validation.csv")

    if cols is None:
        cols = list(tr_dfs[0].columns[1:])

    attacks = va_df.pop("attack")

    cols = ["timestamp"] + cols

    tr_dfs = [df.loc[:, cols] for df in tr_dfs][:]
    va_df = va_df.loc[:, cols]

    df_all = tr_dfs + [va_df]

    for df in df_all:
        drop_cols = [c for c in DROP_COLS if c in cols]
        df.drop(drop_cols, axis=1, inplace=True)

    df_all = pd.concat(df_all).iloc[:, 1:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_all)

    if _type == "train":
        dfs = tr_dfs
    elif _type == "val":
        dfs = [va_df]

    datasets = []
    for df in dfs:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        assert (
            (df - df.shift(1))["timestamp"].dropna() != timedelta(seconds=1)
        ).sum() == 0
        vals = df.iloc[:, 1:].values
        vals = scaler.transform(vals)
        ds = HaiDataset(vals, window_size)
        datasets.append(ds)

    ds = torch.utils.data.ConcatDataset(datasets)

    return ds


# dl = DataLoader(ds, batch_size=64, shuffle=True)
