#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64, Bool
POS_BASE="/pid_pos_"
ROT_BASE="/pid_rot_"
TOPIC="/servo_server/delta_twist_cmds"
ENABLE="/pid_enable_global"
EFFORT="/control_effort"

def read_channel(msg: Float64, variable):
    variable[0] = msg.data

def listener():
    # list to wrap values in reference.
    values = [[0.0] for i in range(6)]
    topics = [POS_BASE+l+EFFORT for l in "xyz"] + [ROT_BASE+l+EFFORT for l in "xyz"]
    rospy.init_node("talker")
    pub = rospy.Publisher(TOPIC,TwistStamped,queue_size=10)
    # to enable the pid controllers
    en = rospy.Publisher(ENABLE,Bool,queue_size=10)
    for val, top in zip(values, topics):
        rospy.Subscriber(top, Float64, read_channel, val)
    update_rate = rospy.Rate(500)
    seq = 0
    while not rospy.is_shutdown():
        en.publish(Bool(True))
        msg = TwistStamped()
        msg.header.seq = seq
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = values[0][0]
        msg.twist.linear.y = values[1][0]
        msg.twist.linear.z = values[2][0]
        msg.twist.angular.x = values[3][0]
        msg.twist.angular.y = values[4][0]
        msg.twist.angular.z = values[5][0]
        pub.publish(msg)
        seq += 1
        update_rate.sleep()



if __name__ == "__main__":
    try:
        listener()
    except rospy.ROSInterruptException():
        pass
