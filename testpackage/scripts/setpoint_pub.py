#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
import rospy
from geometry_msgs.msg import Pose
from tf import TransformListener
SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="base_link"
EE_LINK="ee_link"

def talker():
    topic = SETPOINT_CHANNEL
    pub = rospy.Publisher(topic,Pose,queue_size=10)
    rospy.init_node("talker")
    tf = TransformListener()
    tf.waitForTransform(EE_LINK, WORLDLINK, rospy.Time(), rospy.Duration(1.0))
    dpos, drot = tf.lookupTransform(WORLDLINK, EE_LINK, rospy.Time())
    rospy.loginfo(f"Talker topic: {topic}")
    update_rate = rospy.Rate(10)
    xpos = dpos[0]
    seq = 0
    direction = 1
    while not rospy.is_shutdown():
        if seq % 20 == 0:
            direction *= -1
        xpos = xpos + direction * 0.005
        msg = Pose()
        msg.position.x = xpos
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