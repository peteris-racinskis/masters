#!/usr/bin/env python3
from sys import argv
from bagpy import bagreader
from os.path import exists
import pandas as pd
IFILE="rawdata/recording_with_rotation.bag"
BOTTLE="/mocap_node/Bottle/Odom"
CATCHER="/mocap_node/CatchNet/Odom"
GRIPPER="/mocap_node/TrashPickup/Odom"
NAME_INDEX = 2
TIME="Time"

def odom_extract(df: pd.DataFrame, name: str) -> pd.DataFrame:
    relevant = [
        "Time",
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.position.z",
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "pose.pose.orientation.w",
    ]
    name_map = {s:s.replace("pose.pose",name) for s in relevant[1:]}
    out = df[relevant]
    return out.rename(columns=name_map)

def topic_read(topic: str, bg: bagreader) -> pd.DataFrame:
    name = topic.split("/")[NAME_INDEX]
    topic_csv = bg.message_by_topic(topic)
    topic_df = odom_extract(pd.read_csv(topic_csv), name)
    return topic_df

def df_merge(*dfs) -> pd.DataFrame:
    combined = pd.concat(dfs)
    return combined.sort_values(by=[TIME])

if __name__ == "__main__":
    print("####### STARTING STEP #######")
    print("#######    EXTRACT    #######")
    print("####### STARTING STEP #######")
    fnames = argv[1:] if len(argv) > 1 else [IFILE]
    print(f"Files: {fnames[0]} ... {fnames[-1]}")
    for fname in fnames:
        ofname = fname.replace(".bag", ".csv").replace("rawdata/","processed_data/")
        if not exists(ofname):
            ds = bagreader(fname)
            bottle = topic_read(BOTTLE, ds)
            catch_net = topic_read(CATCHER, ds)
            gripper = topic_read(GRIPPER, ds)
            output = df_merge(bottle, catch_net, gripper)
            output.to_csv(ofname)