from pathlib import Path
from datetime import timedelta

import pandas as pd

def process_df_(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    assert ((df - df.shift(1))['timestamp'].dropna() != timedelta(seconds=1)).sum() == 0

def load_dfs(fns):
    dfs = [pd.read_csv(fn) for fn in fns]
    [process_df_(df) for df in dfs]
    dfs = sorted(dfs, key=lambda df:df.iloc[0]['timestamp'])
    for i in range(len(dfs)):
        if i > 0:
            print(dfs[i].iloc[0]['timestamp'] - dfs[i-1].iloc[-1]['timestamp'])
        print(dfs[i].shape[0], dfs[i].timestamp.min(), "~", dfs[i].timestamp.max())
    return dfs

print("train")
train_dfs = load_dfs(Path("./data/train").glob("*.csv"))
print("val")
val_dfs = load_dfs(Path("./data/validation").glob("*.csv"))
print("test")
test_dfs = load_dfs(Path("./data/test").glob("*.csv"))

print("attach length")
for val_df in val_dfs:
    for k, v in val_df[val_df['attack'] == 1].groupby((val_df['attack'] != 1).cumsum()):
        print(f'[group {k}]', len(v))

def check_cols(dfs):
    df = pd.concat(dfs)
    nu = df.nunique()
    cols_unique = nu[nu == 1].index.tolist()
    cols_binary = nu[nu == 2].index.tolist()
    print(cols_unique)
    print(cols_binary)
    print()

#check_cols(train_dfs)
#check_cols(train_dfs + val_dfs)
#check_cols(train_dfs + val_dfs + test_dfs)

#cols_uniq = []
#bin_cols = ['C08', 'C17', 'C34', 'C46', 'C48', 'C61', 'C64']
drop_cols = ['C34', 'C46', 'C48', 'C61', 'C64']

#assert df.C64.equals(df.C17)
