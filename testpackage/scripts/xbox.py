#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
import rospy
from geometry_msgs.msg import TwistStamped
import pygame
DEF_TOPIC="/testpackage/talker"
TOPIC_KEY="/test_node/talker_topic"

def talker():
    pygame.init()
    joysticks = []
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
        print(f"Detected joystick {joysticks[-1].get_name()}")
    topic = DEF_TOPIC
    params = rospy.get_param_names()
    if TOPIC_KEY in params:
        topic = rospy.get_param(TOPIC_KEY)
    pub = rospy.Publisher(topic,TwistStamped,queue_size=10)
    rospy.init_node("talker")
    rospy.loginfo(f"Talker topic: {topic}")
    update_rate = rospy.Rate(100)
    seq = 0
    while not rospy.is_shutdown():
        pygame.event.get()
        direction_x = joysticks[0].get_axis(0) * 0.2
        direction_y = joysticks[0].get_axis(1) * 0.2
        direction_z = joysticks[0].get_axis(3) * 0.2
        msg = TwistStamped()
        msg.header.seq = seq
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = direction_x
        msg.twist.linear.y = direction_y
        msg.twist.linear.z = direction_z
        pub.publish(msg)
        update_rate.sleep()
        seq += 1

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException():
        pass