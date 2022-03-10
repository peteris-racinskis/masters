#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
import rospy
from geometry_msgs.msg import TwistStamped
DEF_TOPIC="/testpackage/talker"
TOPIC_KEY="/test_node/talker_topic"

def talker():
    topic = DEF_TOPIC
    params = rospy.get_param_names()
    if TOPIC_KEY in params:
        topic = rospy.get_param(TOPIC_KEY)
    pub = rospy.Publisher(topic,TwistStamped,queue_size=10)
    rospy.init_node("talker")
    for p in params:
        rospy.loginfo(f"param name:{p}")
    rospy.loginfo(f"Talker topic: {topic}")
    update_rate = rospy.Rate(10)
    seq = 0
    while not rospy.is_shutdown():
        msg = TwistStamped()
        msg.header.seq = seq
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = -0.1
        msg.twist.angular.y = 0.1
        pub.publish(msg)
        update_rate.sleep()
        seq += 1

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException():
        pass