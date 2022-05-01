#!/usr/bin/env python3
from sre_constants import REPEAT
import pandas as pd
from os import listdir
from os.path import exists
from sys import argv
import random
from typing import List
from hashlib import sha1
TEST_FRAC=0.1
OVERWRITE=True
PREPEND=True
BASE_T="TrashPickup."
BASE_B="Bottle."
RELEASED="Released"
TIME="Time"
NEXT="-n"
INVALID="invalid"
MOVING="Moving"
NOT_DONE="Passed"
POS="position."
ROT="orientation."
DIR="processed_data/norm/"
MODE="start-time"
OFDIR="processed_data/train_datasets/"
IFILE=DIR+"demo-22-03-2022-11:39:49-labelled.csv"
REPEATS=3


# This version does not take velocity into the dataset.
def strip_columns(df: pd.DataFrame, t=False) -> pd.DataFrame:
    # the order of the items in this list determines
    # the order of columbs in the resulting dataframe!
    relevant = [BASE_T+POS+c for c in "xyz"]
    relevant += [BASE_T+ROT+c for c in "xyzw"]
    relevant += [RELEASED]
    relevant += [BASE_B+POS+s for s in ["x-t","y-t","z-t"]]
    if t:
        relevant = [TIME] + relevant
    d = {x:x.replace(BASE_T,"").replace(BASE_B,"") for x in relevant}
    # Keep only the throw. Discard things after and discard the initial wait
    df = df.loc[lambda d: (d[MOVING] == 10) & (d[NOT_DONE] == -10)]
    df = df[relevant].rename(columns=d)
    return df


def prepend_states(df: pd.DataFrame) -> pd.DataFrame:
    row = df.loc[df.index[0]:df.index[0],df.columns]
    for i in range(REPEATS):
        row.index -= 1
        row.loc[:,TIME] = -i - 1
        df = pd.concat([row,df], axis=0)
    return df

def state_transitions(df: pd.DataFrame, t=False) -> pd.DataFrame:
    # Copy over the next values, shift by one, do a row-wise NaN
    # reduction to eliminate invalid rows.
    # DON"T NEED TO REORDER! TARGET COORDINATES ARE AN INPUT PARAMETER,
    # NOT A LABEL YOU FUCKING MORON
    startindex = 1 if t else 0
    shifted = []
    if PREPEND:
        df = prepend_states(df)
    next_states = df[df.columns[startindex:-3]].iloc[1:]
    for i in range(REPEATS):
        next_states.index -= 1
        d = {x:x+NEXT+str(i) for x in next_states.columns}
        col = next_states.rename(columns=d)
        shifted.append(col)
    combined = pd.concat([df]+shifted, axis=1).sort_index()
    combined[INVALID] = combined.isnull().any(axis=1)
    return combined.loc[lambda d: ~d[INVALID]].drop(columns=INVALID)

# For splitting demonstration-wise
def split_train_test(fnames):
    train, test = [], []
    for fname in fnames:
        rnd = random.random()
        if (rnd > TEST_FRAC):
            train.append(fname)
        else:
            test.append(fname)
    return train, test

def process_single_demo(fname: str, start: int, t) -> pd.DataFrame:
    df = state_transitions(strip_columns(pd.read_csv(fname), t), t)
    df.index = range(start + 1, start + len(df) + 1)
    df[RELEASED] = (df[RELEASED].values + 10) / 20
    for i in range(REPEATS):
        df[RELEASED+NEXT+str(i)] = (df[RELEASED+NEXT+str(i)].values + 10) / 20 
    return df

def process_demos_separately(fnames: list, t) -> List[pd.DataFrame]:
    dfs = []
    index = 0
    print("Processing demo list ...")
    for fname in fnames:
        print(fname)
        df_step = process_single_demo(DIR+fname, index, t)
        index = df_step.index[-1]
        dfs.append(df_step)
    return dfs

def process_demo_list(fnames: list, t=False) -> pd.DataFrame:
    df = pd.DataFrame()
    index = 0
    print("Processing demo list ...")
    for fname in fnames:
        print(fname)
        df_step = process_single_demo(DIR+fname, index, t)
        index = df_step.index[-1]
        df = pd.concat([df, df_step])
    return df

def get_dataset_id(train_set, test_set):
    id_base = sha1()
    for s in train_set + test_set:
        id_base.update(bytes(s, encoding='utf_8'))
    return id_base.digest().hex()[:20]


if __name__ == "__main__":
    random.seed(133)
    print("####### STARTING STEP #######")
    print("####### PREPROCESSING #######")
    print("####### STARTING STEP #######")
    mode = argv[1] if len(argv) > 1 else MODE
    t = (mode == "start-time")
    print(f"Normalization mode selected: {mode}")
    fnames = [f for f in filter(lambda x: (f"norm-{mode}" in x), listdir(DIR))]
    train_demos, test_demos = split_train_test(fnames)
    print(f"Demos put into train dataset: {len(train_demos)}")
    print(f"Demos put into test dataset: {len(test_demos)}")
    dataset_id = get_dataset_id(train_demos, test_demos)
    trainname = f"{OFDIR}train-{mode}-doubled-{dataset_id}.csv"
    testname = f"{OFDIR}test-{mode}-doubled-{dataset_id}.csv"
    if PREPEND:
        trainname = trainname.replace(".csv", "-prep.csv")
        testname = testname.replace(".csv", "-prep.csv")
    if not exists(trainname) or OVERWRITE:
        print(f"Creating dataset {trainname}")
        train_df = process_demo_list(train_demos, t)
        train_df.to_csv(trainname, index=False)
    else:
        print(f"Dataset {trainname} already exists! Use OVERWRITE to regenerate")
    if not exists(testname) or OVERWRITE:
        print(f"Creating dataset {testname}")
        test_df = process_demo_list(test_demos, t)
        test_df.to_csv(testname, index=False)
    else:
        print(f"Dataset {testname} already exists! Use OVERWRITE to regenerate")