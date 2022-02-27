import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
IFILE="processed_data/demo-22-02-2022-10:44:48-smooth.csv"
TIME="Time"
BOTTLE="Bottle.position."
GRIPPER="TrashPickup.position."
DISTANCE="Distance"
VELOCITY="Velocity"
ACCELERATION="Acceleration"
THRESHOLD=0.01
WINDOW=5

# Goals:
# 1. detect separation using a differential window (rolling sum before/after)
# 2. fit linear model y=My(t), x=Mx(t)
# 3. fit quadratic model z=Mz(t)
# 4. find intersection Mz(t0) = CatchNet.position.z <- use np.polynomial
# 5. let endpoint = (Mx(t0),My(t0),Mz(t0))

def relative_distance(df: pd.DataFrame) -> pd.DataFrame:
    gripper = []
    bottle =[]
    for axis in "xyz":
        gripper.append(GRIPPER+axis)
        bottle.append(BOTTLE+axis)
    pairs = [list(x) for x in zip(gripper, bottle)]
    dists = []
    for i in range(len(pairs)):
        distance = df[pairs[i]].diff(axis=1)[pairs[i][1]]
        dists.append(distance)
    dists_series = pd.concat(dists, axis=1).pow(2).sum(axis=1).pow(1/2).rename(DISTANCE)
    return pd.concat([df,dists_series],axis=1)

def take_derivative(df: pd.DataFrame, col_in, col_out, window=WINDOW) -> pd.DataFrame:
    forward = df[col_in].rolling(window).sum()
    backward = df[col_in].iloc[::-1].rolling(window).sum()
    velocity = pd.concat([forward,backward],axis=1).diff(axis=1).iloc[:,1].rename(col_out)
    return pd.concat([df,velocity], axis=1)

def relative_velocity(df: pd.DataFrame) -> pd.DataFrame:
    return take_derivative(df, DISTANCE, VELOCITY)

def thresholded_series(length, start):
    off = [-10]*(start)
    on = [10]*(length-start)
    return off + on

def acceleration_thresh(df: pd.DataFrame) -> pd.DataFrame:
    df_a = take_derivative(df, VELOCITY, ACCELERATION)
    valid = df_a.loc[lambda d: pd.notna(d[GRIPPER+"x"])].iloc[2*WINDOW:-2*WINDOW]
    thresh = valid.loc[lambda d: d[ACCELERATION].abs() >= THRESHOLD]
    start = thresh.index[0]
    released = pd.Series(thresholded_series(len(df_a), start), name="Relased")
    return pd.concat([df_a,released], axis=1)



if __name__ == "__main__":
    fnames = sys.argv if len(sys.argv) > 1 else [IFILE]
    for fname in fnames:
        ofname = fname.replace("-smooth","-thresh")
        df = pd.read_csv(IFILE)
        df_d = relative_distance(df)
        df_v = relative_velocity(df_d)
        result = acceleration_thresh(df_v)
        result.to_csv(ofname, index=False)