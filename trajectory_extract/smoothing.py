import pandas as pd
import sys
IFILE="processed_data/demo-22-02-2022-10:44:48.csv"
OFFSET=1
WINDOW=20


if __name__ == "__main__":
    fnames = sys.argv if len(sys.argv) > 1 else [IFILE]
    for fname in fnames:
        ofname = f"{fname[:-4]}-smooth.csv"
        df = pd.read_csv(fname)
        smoothed = df[df.columns[OFFSET:]].rolling(WINDOW, center=True).mean()
        unchanged = df[df.columns[:OFFSET]]
        result = pd.concat([unchanged, smoothed], axis=1)
        result.to_csv(ofname, index=False)