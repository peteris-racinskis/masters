#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import pandas as pd
from tf import TransformListener
SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="base_link"
EE_LINK="ee_link"
IFILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal-0-first-attempt.csv"

def talker(df: pd.DataFrame):
    topic = SETPOINT_CHANNEL
    pub = rospy.Publisher(topic,PoseStamped,queue_size=10)
    rospy.init_node("talker")
    #tf = TransformListener()
    #tf.waitForTransform(EE_LINK, WORLDLINK, rospy.Time(), rospy.Duration(1.0))
    #dpos, drot = tf.lookupTransform(WORLDLINK, EE_LINK, rospy.Time())
    now = rospy.get_rostime()
    rospy.loginfo(f"Talker topic: {topic}")
    update_rate = rospy.Rate(1000)
    seq = 0
    ind = 0
    direction = 1
    while not rospy.is_shutdown():
        dpos = df[["x","y","z"]].values[ind]
        drot = df[["rx","ry","rz", "rw"]].values[ind]
        stamp = df["Time"].values[ind]
        secs = int(stamp)
        nsecs = int(1e9*(stamp-secs))
        delta = rospy.Duration(secs,nsecs)
        t = now + delta
        #if ind in [0,49] and seq != 0:
            #direction = -direction
        if ind == 49:
            break
        ind += direction
        msg = PoseStamped()
        msg.header.seq = seq
        msg.header.stamp.secs = t.secs
        msg.header.stamp.nsecs = t.nsecs
        msg.pose.position.x = dpos[0]
        msg.pose.position.y = dpos[1]
        msg.pose.position.z = dpos[2]
        msg.pose.orientation.x = drot[0]
        msg.pose.orientation.y = drot[1]
        msg.pose.orientation.z = drot[2]
        msg.pose.orientation.w = drot[3]
        pub.publish(msg)
        update_rate.sleep()
        seq += 1

if __name__ == "__main__":
    df = pd.read_csv(IFILE).iloc[:50]
    talker(df)