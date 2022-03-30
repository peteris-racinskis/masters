from posixpath import split
import pandas as pd
from os import listdir
import random
TEST_FRAC=0.1
random.seed(133)
BASE_T="TrashPickup."
BASE_B="Bottle."
RELEASED="Released"
POS="position."
ROT="orientation."
IFILE="processed_data/labelled/demo-22-03-2022-11:39:49-labelled.csv"

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    # the order of the items in this list determines
    # the order of columbs in the resulting dataframe!
    relevant = [BASE_T+POS+c for c in "xyz"]
    relevant += [BASE_T+ROT+c for c in "xyzw"]
    relevant += [RELEASED]
    relevant += [BASE_B+POS+s for s in ["x-t","y-t","z-t"]]
    d = {x:x.replace(BASE_T,"").replace(BASE_B,"") for x in relevant}
    df = df[relevant].rename(columns=d)
    return df

def state_transitions(df: pd.DataFrame) -> pd.DataFrame:
    # Copy over the next values, shift by one, do a row-wise NaN
    # reduction to eliminate invalid rows, reorder.
    next_states = df[df.columns[:-3]].iloc[1:]
    next_states.index -= 1
    d = {x:x+"-next" for x in next_states.columns}
    next_states = next_states.rename(columns=d)
    combined = pd.concat([df,next_states], axis=1)
    combined["invalid"] = combined.isnull().any(axis=1)
    reorder = list(df.columns[:-3]) + list(next_states.columns) + list(df.columns[-3:])
    return combined.loc[lambda d: ~d["invalid"]][reorder]

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

def process_single_demo(fname: str) -> pd.DataFrame:
    return state_transitions(strip_columns(pd.read_csv(fname)))


if __name__ == "__main__":
    fnames = listdir("processed_data/labelled")
    train_demos, test_demos = split_train_test(fnames)
    print(f"Demos put into train dataset: {len(train_demos)}")
    [print(x) for x in train_demos]
    print(f"Demos put into test dataset: {len(test_demos)}")
    [print(x) for x in test_demos]
    df = pd.read_csv(IFILE)
    df = strip_columns(df)
    df = state_transitions(df)
    pass