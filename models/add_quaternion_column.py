import pandas as pd
from os import listdir
from helpers import quaternion_norm
DIR="models/"


if __name__ == "__main__":
    fnames = [DIR+f for f in filter(lambda x: (".csv" in x), listdir(DIR))]
    for fname in fnames:
        try:
            df = pd.read_csv(fname)
            df = quaternion_norm(df)
            df.to_csv(fname.replace(".csv", "-qn.csv"), index=False)
        except Exception as e:
            print(f"Failed on {fname} - {repr(e)}")
            pass