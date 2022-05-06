#!/usr/bin/env python3
import pandas as pd
from datetime import datetime
from typing import List
from sys import argv
IFILE="processed_data/regular_timeseries.csv"
OFILE_BASE="processed_data/split/demo-"
TIME="Time"
GRIPPER_X="TrashPickup.position.x"
GRIPPER_XLIMU="TrashPickup.position.xlim+"
GRIPPER_XLIML="TrashPickup.position.xlim-"
BOTTLE_X="Bottle.position.x"
BOTTLE_XLIMU="Bottle.position.xlim+"
BOTTLE_XLIML="Bottle.position.xlim-"
WINDOW=80
STEPS=150
# Set this for every batch of demonstrations in the makefile
BOX_X_H = 1.0 # This is for recording_with_rotation.bag
BOX_X_L = 0.9 # This is for recording_with_rotation.bag

def demo_starts(df: pd.DataFrame, box_h, box_l) -> pd.DataFrame:
    upper = df.rolling(150).max()[[GRIPPER_X, BOTTLE_X]].rename(columns={GRIPPER_X:GRIPPER_XLIMU, BOTTLE_X:BOTTLE_XLIMU})
    lower = df.rolling(150).min()[[GRIPPER_X, BOTTLE_X]].rename(columns={GRIPPER_X:GRIPPER_XLIML, BOTTLE_X:BOTTLE_XLIML})
    combined = pd.concat([df, upper, lower], axis=1)
    starts = combined.loc[lambda d: (d[GRIPPER_XLIMU] <= box_h) &
                                    (d[GRIPPER_XLIML] >= box_l) &
                                    (d[BOTTLE_XLIMU] <= box_h) &
                                    (d[BOTTLE_XLIML] >= box_l)]
    diffs = starts[TIME].diff()
    return diffs.loc[lambda d: d > 5].index

def subframes(df: pd.DataFrame, indices) -> List[pd.DataFrame]:
    dfs = []
    for index in indices:
        dfs.append(df.iloc[index:index+STEPS])
    return dfs

def name_and_write(df: pd.DataFrame):
    t = df.iloc[0][TIME]
    timestamp = datetime.utcfromtimestamp(t).strftime('%d-%m-%Y-%H:%M:%S')
    fname = f"{OFILE_BASE}{timestamp}.csv"
    df.to_csv(fname, index=False)

# CALL SYNTAX:
# tracjectory_extract/split.py <limit low> <limit high> <filenames>

if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######     SPLIT     #######")
    print("####### STARTING STEP #######")
    fnames = argv[3:] if len(argv) > 3 else [IFILE]
    print(f"Files: {fnames[0]} ... {fnames[-1]}")
    box_h, box_l = [float(x) for x in argv[1:3]] if len(argv) > 3 else [BOX_X_H, BOX_X_L]
    for fname in fnames:
        df = pd.read_csv(fname)
        start_indices = demo_starts(df, box_h, box_l)
        demos = subframes(df, start_indices)
        for demo in demos:
            name_and_write(demo)
