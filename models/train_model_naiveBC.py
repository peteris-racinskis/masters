import pandas as pd
import numpy as np
from os import listdir
from typing import Tuple
TRAIN="processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
TEST="processed_data/train_datasets/test-003430811ff20c35ccd5.csv"

def data_and_label(df: pd.DataFrame) -> Tuple[np.ndarray]:
    data_cols = df.columns[:-8]
    label_cols = df.columns[-8:]
    return df[data_cols].values, df[label_cols].values

train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
test_data, test_labels = data_and_label(pd.read_csv(TEST))
pass