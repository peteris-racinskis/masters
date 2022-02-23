from bagpy import bagreader
import pandas as pd
IFILE="rawdata/record_all_demonstrations.bag"
OFILE="processed_data/combined_timeseries.csv"
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
    ds = bagreader(IFILE)
    bottle = topic_read(BOTTLE, ds)
    catch_net = topic_read(CATCHER, ds)
    gripper = topic_read(GRIPPER, ds)
    output = df_merge(bottle, catch_net, gripper)
    output.to_csv(OFILE)