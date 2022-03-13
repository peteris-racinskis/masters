#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
import rospy
from geometry_msgs.msg import Pose
SETPOINT_CHANNEL="/pid_setpoint"

def talker():
    topic = SETPOINT_CHANNEL
    dpos = [0.58, -0.24, 0.43]
    drot = [0.7895978776147842, 0, 0, 1]
    pub = rospy.Publisher(topic,Pose,queue_size=10)
    rospy.init_node("talker")
    rospy.loginfo(f"Talker topic: {topic}")
    update_rate = rospy.Rate(10)
    seq = 0
    while not rospy.is_shutdown():
        msg = Pose()
        msg.position.x = dpos[0]
        msg.position.y = dpos[1]
        msg.position.z = dpos[2]
        msg.orientation.x = drot[0]
        msg.orientation.y = drot[1]
        msg.orientation.z = drot[2]
        msg.orientation.w = drot[3]
        pub.publish(msg)
        update_rate.sleep()
        seq += 1

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException():
        pass