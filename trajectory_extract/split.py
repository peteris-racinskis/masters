from os import times
import pandas as pd
from datetime import datetime
from typing import List
IFILE="processed_data/regular_timeseries.csv"
OFILE_BASE="processed_data/demo-"
TIME="Time"
GRIPPER_X="TrashPickup.position.x"
GRIPPER_XLIMU="TrashPickup.position.xlim+"
GRIPPER_XLIML="TrashPickup.position.xlim-"
BOTTLE_X="Bottle.position.x"
BOTTLE_XLIMU="Bottle.position.xlim+"
BOTTLE_XLIML="Bottle.position.xlim-"
WINDOW=150
STEPS=500

def demo_starts(df: pd.DataFrame) -> pd.DataFrame:
    upper = df.rolling(150).max()[[GRIPPER_X, BOTTLE_X]].rename(columns={GRIPPER_X:GRIPPER_XLIMU, BOTTLE_X:BOTTLE_XLIMU})
    lower = df.rolling(150).min()[[GRIPPER_X, BOTTLE_X]].rename(columns={GRIPPER_X:GRIPPER_XLIML, BOTTLE_X:BOTTLE_XLIML})
    combined = pd.concat([df, upper, lower], axis=1)
    starts = combined.loc[lambda d: (d[GRIPPER_XLIMU] <= 1.2) &
                                    (d[GRIPPER_XLIML] >= 1.1) &
                                    (d[BOTTLE_XLIMU] <= 1.2) &
                                    (d[BOTTLE_XLIML] >= 1.1)]
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
    df.to_csv(fname)


if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    start_indices = demo_starts(df)
    demos = subframes(df, start_indices)
    for demo in demos:
        name_and_write(demo)

