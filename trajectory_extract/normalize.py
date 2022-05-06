#!/usr/bin/env python3
from os.path import exists
from sys import argv

import pandas as pd
import numpy as np

OVERWRITE=True
BOTTLE="Bottle.position."
TIME="Time"
GRIPPER="TrashPickup.position."
IFILE="processed_data/labelled/demo-22-02-2022-10:44:48-labelled.csv"
MODE="start-time"
explain = {
    "default":  "Do nothing.",
    "start": "Center trajectory on the first moving entry.",
    "start-time": "Center trajectory on the first moving entry. Normalize time too",
    "target": "Center trajectory on the target coordinates"
}

def normalize_to_target(df: pd.DataFrame) -> pd.DataFrame:
    t_cols = [BOTTLE+c+"-t" for c in "xyz"]
    cols = [GRIPPER+c for c in "xyz"]
    target_values = {c:df[tc][0] for c,tc in zip(cols,t_cols)}
    for col in cols:
        df[col] = df[col] - target_values[col]
    for tcol in t_cols:
        df[tcol] = 0.0
    return df

def normalize_to_start(df: pd.DataFrame, t=False) -> pd.DataFrame:
    start_index = df.loc[lambda d: d["Moving"] == 10].index[0]
    cols = [GRIPPER+c for c in "xyz"]
    t_cols = [BOTTLE+c+"-t" for c in "xyz"]
    start_values = {c:df[c][start_index] for c in cols}
    start_time = df[TIME][start_index]
    df[TIME] = df[TIME] - start_time
    for col, tcol in zip(cols, t_cols):
        df[col] = df[col] - start_values[col]
        df[tcol] = df[tcol] - start_values[col]
    return df


if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######   NORMALIZE   #######")
    print("####### STARTING STEP #######")
    mode = argv[1] if len(argv) > 1 else MODE
    fnames = argv[2:] if len(argv) > 2 else [IFILE]
    print(f"Mode: {mode}. {explain[mode]}")
    for fname in fnames:
        if not "demo-" in fname:
            continue
        ofname = fname.replace("labelled/demo", "norm/demo")
        ofname = ofname.replace("-labelled", f"-norm-{mode}")
        if not exists(ofname) or OVERWRITE:
            print(fname)
            df = pd.read_csv(fname)
            if mode == "start":
                df = normalize_to_start(df)
            if mode == "start-time":
                df = normalize_to_start(df, True)
            elif mode == "target":
                df = normalize_to_target(df)
            df.to_csv(ofname, index=False)
