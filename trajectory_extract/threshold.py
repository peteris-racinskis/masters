#!/usr/bin/env python3
import pandas as pd
import sys
from os.path import exists
IFILE="processed_data/smoothed/demo-22-02-2022-10:44:48-smooth.csv"
TIME="Time"
BOTTLE="Bottle.position."
GRIPPER="TrashPickup.position."
CATCHPOS="CatchNet.position.x"
DISTANCE="Distance"
VELOCITY_REL="Velocity_rel"
VELOCITY_ABS="Velocity_abs"
ACCELERATION="Acceleration"
RELEASED="Released"
FREEFALL="Freefall"
PASSED="Passed"
MOVING="Moving"
THRESHOLD_RELEASE=0.05
THRESHOLD_FREEFALL=0.1
THRESHOLD_MOVING=0.2
WINDOW=5
OVERWRITE=True
SAMPLE_RATE=100

# Goals:
# 1. detect separation using a differential window (rolling sum before/after)

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
    backward = df[col_in].iloc[::-1].rolling(window).sum().iloc[::-1] # NEED TO REVERSE AGAIN!!!
    velocity = pd.concat([forward,backward],axis=1).diff(axis=1).iloc[:,1].rename(col_out)
    return pd.concat([df,velocity], axis=1)

def relative_velocity(df: pd.DataFrame) -> pd.DataFrame:
    return take_derivative(df, DISTANCE, VELOCITY_REL)

# Gripper absolute velocity
def absolute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for axis in "xyz":
        cols.append(GRIPPER + axis)
    d = {x:x.replace("position", "velocity") for x in cols}
    velocities = df[cols].diff().rename(columns=d).mul(SAMPLE_RATE)
    velocity = velocities.pow(2).sum(axis=1).pow(1/2)
    velocity.name = VELOCITY_ABS
    return pd.concat([df,velocities,velocity], axis=1)

def add_thresholded_series(df, thresh, name):
    length = len(df)
    start = thresh.index[0]
    off = [-10]*(start)
    on = [10]*(length-start)
    return pd.concat([df,pd.Series(off+on, name=name)],axis=1)

def acceleration_thresh(df: pd.DataFrame) -> pd.DataFrame:
    df_a = take_derivative(df, VELOCITY_REL, ACCELERATION)
    valid = df_a.loc[lambda d: pd.notna(d[GRIPPER+"x"])].iloc[2*WINDOW:-2*WINDOW]
    thresh_r = valid.loc[lambda d: d[ACCELERATION].abs() >= THRESHOLD_RELEASE]
    thresh_f = valid.loc[lambda d: d[VELOCITY_REL].abs() >= THRESHOLD_FREEFALL]
    thresh_v = valid.loc[lambda d: d[VELOCITY_ABS].abs() >= THRESHOLD_MOVING]
    df_r = add_thresholded_series(df_a, thresh_r, RELEASED)
    df_v = add_thresholded_series(df_r, thresh_v, MOVING)
    return add_thresholded_series(df_v, thresh_f, FREEFALL)

def position_thresh(df: pd.DataFrame) -> pd.DataFrame:
    catchpos = df[CATCHPOS].mean()
    bx = BOTTLE+"x"
    thresh = df.loc[lambda d: d[bx] <= catchpos]
    return add_thresholded_series(df, thresh, PASSED)



if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######   THRESHOLD   #######")
    print("####### STARTING STEP #######")
    fnames = sys.argv[1:] if len(sys.argv) > 1 else [IFILE]
    print(f"Files: {fnames[0]} ... {fnames[-1]}")
    for fname in fnames:
        if not "demo-22-0" in fname:
            continue
        ofname = fname.replace("/smoothed/demo", "/thresh/demo")
        ofname = ofname.replace("-smooth","-thresh")
        if not exists(ofname) or OVERWRITE:
            df = pd.read_csv(fname)
            try:
                df_av = absolute_velocity(df)
                df_d = relative_distance(df_av)
                df_v = relative_velocity(df_d)
                released = acceleration_thresh(df_v)
                passed = position_thresh(released)
                print(fname)
            except Exception as e:
                print(f"Failed on {fname}")
                continue
            passed.to_csv(ofname, index=False)