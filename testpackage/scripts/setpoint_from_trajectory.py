#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
from black import out
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory
import pandas as pd
import numpy as np
import copy
import sys
import moveit_commander
from tf.transformations import quaternion_inverse, quaternion_multiply
from typing import List
from tensorflow.keras import models

SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="world"
EE_LINK="panda_link8"
IFILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal-0-first-attempt.csv"
MODEL_FILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal"
BASE_ROT=np.asarray([0.023,-0.685,0.002,-0.729])
FRAME_CORR=np.asarray([0,-1,0,1])
TARGET_COORDS=np.asarray([-2.59361591383002,-0.178823692236341,-0.36553905703157])


def point_from_pose(pose: Pose) -> List:
    p = pose.position
    return [p.x, p.y, p.z]

def quat_from_pose(pose: Pose) -> List:
    o = pose.orientation
    return [o.x, o.y, o.z, o.w]

def pose_from_point(pose: Pose, point) -> Pose:
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]

def pose_from_quat(pose: Pose, quat) -> Pose:
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

# need to map whatever the default rotation of the gripper is
# a roughly centered rotation within the demonstration data
# distribution. then apply the same shift to everything in 
# the actual robot reference frame
# q1 - roughly centered rot in demo dataset
# q2 - initial state of the robot gripper
# p - transform
# find according to:
# q2 = q1*p'
# p*q2 = p*q1*p' = q1
# q2'*p*q2 = p = q2'*q1
def find_rot_offset(first_rot):
    frame_shifted = quaternion_multiply(normalize(FRAME_CORR), first_rot)
    inv = quaternion_inverse(frame_shifted)
    offs = normalize(quaternion_multiply(inv, normalize(BASE_ROT)))
    return offs, quaternion_inverse(offs)

def rot_comp(offs, rot):
    return quaternion_multiply(offs, rot)

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector ** 2))

def msg_from_model(model):
    msgs = []
    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    rot_offset, rot_restore = find_rot_offset(quat_from_pose(msg))
    pos_offset = np.asarray(point_from_pose(msg))
    output_state = np.asarray([0.0]*3 + list(rot_comp(rot_offset, quat_from_pose(msg))) + [0]).reshape(1,-1)
    t_base = np.asarray([0.01]).reshape(1,-1)
    for i in range(64):
        msgs.append(msg)
        t = t_base * i
        output_state = output_state.astype(np.float32)
        state = np.concatenate([t, output_state, np.asarray(TARGET_COORDS).reshape(1,-1)], axis=1)
        output_state = model(state).numpy().astype(np.float64)
        output_state[0,3:7] = normalize(output_state[0,3:7])
        msg = Pose()
        pose_from_point(msg, output_state[0,(1,0,2)] * 0.3 + pos_offset)
        #pose_from_quat(msg, init_rot)
        pose_from_quat(msg, rot_comp(output_state[0,3:7], rot_restore))
    return msgs

def int_time_to_float(secs, nsecs):
    return secs + 1e-9 * nsecs

def float_time_to_int(time):
    secs = int(time)
    nsecs = int(1e9*(time-secs))
    return secs, nsecs

def rescale_time(trajectory: JointTrajectory, dt=1.0):
    finish_time_int = trajectory.points[-1].time_from_start
    finish_time = int_time_to_float(finish_time_int.secs, finish_time_int.nsecs)
    scaler = dt / finish_time
    for point in trajectory.points:
        newtime_int = point.time_from_start
        newtime = scaler * int_time_to_float(newtime_int.secs, newtime_int.nsecs)
        point.time_from_start.secs, point.time_from_start.nsecs = float_time_to_int(newtime)
    return trajectory


def execute_trajectory(df: pd.DataFrame):
    topic = SETPOINT_CHANNEL
    #pub = rospy.Publisher(topic,PoseStamped,queue_size=10)
    #rospy.init_node("talker")
    #tf = TransformListener()
    #tf.waitForTransform(EE_LINK, WORLDLINK, rospy.Time(), rospy.Duration(1.0))
    #dpos, drot = tf.lookupTransform(WORLDLINK, EE_LINK, rospy.Time())
    now = rospy.get_rostime()
    #current_pose = group.get_current_pose()
    rospy.loginfo(f"Talker topic: {topic}")
    model = models.load_model(MODEL_FILE)
    msgs = msg_from_model(model)
    p,f = group.compute_cartesian_path([x for x in msgs], 0.1, 0.0)
    p.joint_trajectory = rescale_time(p.joint_trajectory, 20)
    for pp in p.joint_trajectory.points:
        pp.velocities = []
        pp.accelerations = []
        #pp.time_from_start.secs += 1
    group.execute(p, wait=True)

if __name__ == "__main__":
    df = pd.read_csv(IFILE).iloc[:50]
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    group = moveit_commander.MoveGroupCommander(group_name)
    planning_frame = group.get_planning_frame()
    print(f"============ Reference frame: {planning_frame}")
    eef_link = group.get_end_effector_link()
    print(f"============ End effector link: {eef_link}")
    group_names = robot.get_group_names()
    print(f"============ Available Planning Groups: {robot.get_group_names()}")

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")
    execute_trajectory(df)