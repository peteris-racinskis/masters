#!/usr/bin/env python3
import rospy
import sys
from typing import List
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_inverse, quaternion_multiply
from tf import TransformListener
BASE_TOPIC="pid_error_terms/"
BASE_POS=BASE_TOPIC+"position_"
BASE_ROT=BASE_TOPIC+"rotation_"
WORLDLINK="world"
EE_LINK="ee_link"
NODENAME="tf_to_pose_error"
SETPOINT_CHANNEL="/pid_setpoint"

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
    lst, base, target, pnodes, qnodes = cb_args
    posgoal = point_from_pose(msg)
    rotgoal = quat_from_pose(msg)
    pos, rot = lst.lookupTransform(base, target, rospy.Time())
    rot_err = quaternion_multiply(rotgoal, quaternion_inverse(rot))
    rot_err_eul = list(euler_from_quaternion(rot_err))
    poserr = lsub(posgoal, pos)
    for node, value in zip(pnodes + qnodes, poserr + rot_err_eul):
        m = Float64(value)
        node.publish(m)

def listener():
    letters = "xyzw"
    pnodes = []
    qnodes = []
    for pos in letters[:-1]:
        pnodes.append(rospy.Publisher(BASE_POS+pos, Float64, queue_size=10))
    for rot in letters:
        qnodes.append(rospy.Publisher(BASE_ROT+rot, Float64, queue_size=10))
    rospy.init_node(NODENAME)
    tf = TransformListener()
    rospy.Subscriber(SETPOINT_CHANNEL,
                     Pose, 
                     publish_error_msg, 
                     callback_args=(tf, WORLDLINK, EE_LINK, pnodes, qnodes))
    rospy.spin()


if __name__ == "__main__":
    try:
        listener()
    except rospy.ROSInterruptException():
        pass