#!/usr/bin/env python3
from encodings import utf_8
import pandas as pd
from os import listdir
from os.path import exists
from sys import argv
import random
from hashlib import sha1
TEST_FRAC=0.1
OVERWRITE=True
BASE_T="TrashPickup."
BASE_B="Bottle."
RELEASED="Released"
NEXT="-next"
INVALID="invalid"
MOVING="Moving"
NOT_DONE="Passed"
POS="position."
ROT="orientation."
DIR="processed_data/norm/"
MODE="start"
OFDIR="processed_data/train_datasets/"
IFILE=DIR+"demo-22-03-2022-11:39:49-labelled.csv"


# This version does not take velocity into the dataset.
def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    # the order of the items in this list determines
    # the order of columbs in the resulting dataframe!
    relevant = [BASE_T+POS+c for c in "xyz"]
    relevant += [BASE_T+ROT+c for c in "xyzw"]
    relevant += [RELEASED]
    relevant += [BASE_B+POS+s for s in ["x-t","y-t","z-t"]]
    d = {x:x.replace(BASE_T,"").replace(BASE_B,"") for x in relevant}
    # Keep only the throw. Discard things after and discard the initial wait
    df = df.loc[lambda d: (d[MOVING] == 10) & (d[NOT_DONE] == -10)]
    df = df[relevant].rename(columns=d)
    return df

def state_transitions(df: pd.DataFrame) -> pd.DataFrame:
    # Copy over the next values, shift by one, do a row-wise NaN
    # reduction to eliminate invalid rows.
    # DON"T NEED TO REORDER! TARGET COORDINATES ARE AN INPUT PARAMETER,
    # NOT A LABEL YOU FUCKING MORON
    next_states = df[df.columns[:-3]].iloc[1:]
    next_states.index -= 1
    d = {x:x+NEXT for x in next_states.columns}
    next_states = next_states.rename(columns=d)
    combined = pd.concat([df,next_states], axis=1)
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

def process_single_demo(fname: str, start: int) -> pd.DataFrame:
    df = state_transitions(strip_columns(pd.read_csv(fname)))
    df.index = range(start + 1, start + len(df) + 1)
    df[RELEASED] = (df[RELEASED].values + 10) / 20 
    df[RELEASED+NEXT] = (df[RELEASED+NEXT].values + 10) / 20 
    return df

def process_demo_list(fnames: list) -> pd.DataFrame:
    df = pd.DataFrame()
    index = 0
    print("Processing demo list ...")
    for fname in fnames:
        print(fname)
        df_step = process_single_demo(DIR+fname, index)
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
    print(f"Normalization mode selected: {mode}")
    fnames = [f for f in filter(lambda x: (f"norm-{mode}" in x), listdir(DIR))]
    train_demos, test_demos = split_train_test(fnames)
    print(f"Demos put into train dataset: {len(train_demos)}")
    print(f"Demos put into test dataset: {len(test_demos)}")
    dataset_id = get_dataset_id(train_demos, test_demos)
    trainname = f"{OFDIR}train-{mode}-{dataset_id}.csv"
    testname = f"{OFDIR}test-{mode}-{dataset_id}.csv"
    if not exists(trainname) or OVERWRITE:
        print(f"Creating dataset {trainname}")
        train_df = process_demo_list(train_demos)
        train_df.to_csv(trainname, index=False)
    else:
        print(f"Dataset {trainname} already exists! Use OVERWRITE to regenerate")
    if not exists(testname) or OVERWRITE:
        print(f"Creating dataset {testname}")
        test_df = process_demo_list(test_demos)
        test_df.to_csv(testname, index=False)
    else:
        print(f"Dataset {testname} already exists! Use OVERWRITE to regenerate")