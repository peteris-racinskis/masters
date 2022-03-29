#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sys import argv
from typing import Tuple
IFILE="processed_data/combined_timeseries.csv"
OFILE="processed_data/regular_timeseries.csv"
TIME="Time"

def trim_float(f, step):
    inv = 1 / step
    return float((int(f * inv) + 1)) / inv

def resample(df: pd.DataFrame, step=0.01) -> Tuple[pd.DataFrame, np.ndarray]:
    start = trim_float(df.iloc[0][TIME], step)
    end = trim_float(df.iloc[-1][TIME], step)
    times = np.arange(start=start, stop=end, step=step)
    d={"Time": times, "Unnamed: 0": "Not a number"}
    cols = df.columns
    regular = pd.DataFrame(data=d, columns=cols)
    return regular, times

def interpolate(df: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df, reg])
    combined: pd.DataFrame = combined.drop_duplicates(subset=[TIME]).sort_values(by=[TIME])
    interp = combined.fillna(method="ffill")
    return interp.loc[lambda d: d["Unnamed: 0"] == "Not a number"]


if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######  REGULARIZE   #######")
    print("####### STARTING STEP #######")
    fnames = argv[1:] if len(argv) > 1 else [IFILE]
    print(f"Files: {fnames[0]} ... {fnames[-1]}")
    for fname in fnames:
        df = pd.read_csv(fname)
        ofname = fname.replace(".csv", "-regularized.csv")
        rows, index = resample(df)
        interpolated = interpolate(df, rows)
        sel = interpolated.columns[1:]
        interpolated[sel].to_csv(ofname, index=False)