import rospy
import tensorflow as tf
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose


REQUEST_CHANNEL="/model_requests"
MODEL="models/naiveBC-norm-start-timesignal"

def publish_next_state(msg: Pose, cb_args):
    # get latest
    pass

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
    rospy.Subscriber(REQUEST_CHANNEL,
                     Pose, 
                     publish_next_state, 
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