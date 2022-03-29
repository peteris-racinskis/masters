import pandas as pd
import sys
IFILE="processed_data/labelled/demo-22-02-2022-10:44:48-labelled.csv"
NET="TrashPickup.position."
NTX = NET+"x"
NTY = NET+"y"
NTZ = NET+"z"
TIME="Time"

def all_derivatives(df: pd.DataFrame, window=5) -> pd.DataFrame:
    forward = df.rolling(window).sum()
    backward = df.iloc[::-1].rolling(window).sum().iloc[::-1]
    dt_inv = 1 / (window * (df["Time"].iloc[1] - df["Time"].iloc[0]))
    diff = (backward - forward.values) * dt_inv
    return diff[[NTX,NTY,NTZ]]


if __name__ == "__main__":
    fnames = sys.argv[1:] if len(sys.argv) > 1 else [IFILE]
    for fname in fnames:
        if not "demo-22-02" in fname:
            continue
        print(fname)
        ofname = fname.replace("/labelled/demo", "/vel/demo")
        ofname = ofname.replace("-labelled","-vel")
        df = pd.read_csv(fname)
        v = all_derivatives(df)
        v.to_csv(ofname, index=False)