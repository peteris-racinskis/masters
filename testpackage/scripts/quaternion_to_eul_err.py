#!/usr/bin/env python3
import rospy
import sys
from typing import List
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_inverse, quaternion_multiply
from tf import TransformListener
BASE_SET_TOPIC="pid_setpoint_terms/"
BASE_CUR_TOPIC="pid_state_terms/"
BASE_POS_SET=BASE_SET_TOPIC+"position_"
BASE_POS_CUR=BASE_CUR_TOPIC+"position_"
BASE_ROT_SET=BASE_SET_TOPIC+"rotation_"
BASE_ROT_CUR=BASE_CUR_TOPIC+"rotation_"
WORLDLINK="base_link"
EE_LINK="ee_link"
NODENAME="tf_to_pose_error"
SETPOINT_CHANNEL="/pid_setpoint"


'''
 What this script does:
 1. Extract position, orientation from a pose message;
 2. Compute quaternion error, convert to euler angle;
 3. Publish the current position, orientation error;
 4. Publish the setpoint position;
 5. Orientation is set to 0, the error term is what matters.
'''

def point_from_pose(pose: Pose) -> List:
    p = pose.position
    return [p.x, p.y, p.z]

def quat_from_pose(pose: Pose) -> List:
    o = pose.orientation
    return [o.x, o.y, o.z, o.w]

def lsub(l1, l2):
    return [ll1-ll2 for (ll1,ll2) in zip(l1,l2)]

def publish_error_msg(msg: Pose, cb_args):
    # get latest
    lst, base, target, pnodes_g, pnodes_c, qnodes_g, qnodes_c = cb_args
    posgoal = point_from_pose(msg)
    rotgoal = quat_from_pose(msg)
    pos, rot = lst.lookupTransform(base, target, rospy.Time())
    rot_err = quaternion_multiply(rotgoal, quaternion_inverse(rot))
    rot_err_eul = list(euler_from_quaternion(rot_err))
    # let angle error setpoint = 0,0,0;
    # then the controller will seek to drive it to 0.
    for node, value in zip(
         pnodes_g + pnodes_c + qnodes_g + qnodes_c, 
         posgoal + pos  + [0,0,0] + rot_err_eul):
        m = Float64(value)
        node.publish(m)

def listener():
    letters = "xyz"
    pnodes_g = []
    pnodes_c = []
    qnodes_g = []
    qnodes_c = []
    # I don't have quaternions to publish duh
    # the angle error setpoint is constant and the angle error is in 
    # euler, not quaternion form.
    rospy.init_node("talker")
    for pos in letters:
        pnodes_g.append(rospy.Publisher(BASE_POS_SET+pos, Float64, queue_size=10))
        pnodes_c.append(rospy.Publisher(BASE_POS_CUR+pos, Float64, queue_size=10))
        qnodes_g.append(rospy.Publisher(BASE_ROT_SET+pos, Float64, queue_size=10))
        qnodes_c.append(rospy.Publisher(BASE_ROT_CUR+pos, Float64, queue_size=10))
    tf = TransformListener()
    rospy.Subscriber(SETPOINT_CHANNEL,
                     Pose, 
                     publish_error_msg, 
                     callback_args=(
                         tf, 
                         WORLDLINK, 
                         EE_LINK, 
                         pnodes_g,  
                         pnodes_c,
                         qnodes_g,
                         qnodes_c))
    rospy.spin()


if __name__ == "__main__":
    try:
        listener()
    except rospy.ROSInterruptException():
        pass