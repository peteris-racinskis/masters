#!/usr/bin/env python3
import pandas as pd
import sys
from os.path import exists
IFILE="processed_data/demo-22-02-2022-10:44:48.csv"
OFFSET=1
WINDOW=20
OVERWRITE=False


if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######   SMOOTHING   #######")
    print("####### STARTING STEP #######")
    fnames = sys.argv[1:] if len(sys.argv) > 1 else [IFILE]
    print(f"Files: {fnames[0]} ... {fnames[-1]}")
    for fname in fnames:
        if not "demo-22-0" in fname:
            continue
        ofname = fname.replace("/demo", "/smoothed/demo")
        ofname = f"{ofname[:-4]}-smooth.csv"
        if not exists(ofname) or OVERWRITE:
            print(fname)
            df = pd.read_csv(fname)
            smoothed = df[df.columns[OFFSET:]].rolling(WINDOW, center=True).mean()
            unchanged = df[df.columns[:OFFSET]]
            result = pd.concat([unchanged, smoothed], axis=1)
            result.to_csv(ofname, index=False)