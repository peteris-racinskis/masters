#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
from ntpath import join
from black import out
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory
import pandas as pd
import numpy as np
import copy
import sys
import moveit_commander
from tf.transformations import quaternion_inverse as qi, quaternion_multiply as qm
from typing import List
from tensorflow.keras import models

SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="world"
EE_LINK="panda_link8"
IFILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal-0-first-attempt.csv"
IFILE="/home/user/repos/masters/processed_data/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"
MODEL_FILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal"
MODEL_FILE="/home/user/repos/masters/models/BCO-256x2-256x2-start-timesignal-doubled-noreg-ep200-b64-norm-gen"
BASE_ROT=np.asarray([0.023,-0.685,0.002,-0.729])
#BASE_ROT=np.asarray([ 0.46312436,  0.51316392, -0.48667049,  0.52832292])
JOINT_GOAL=[1.577647875986087, 0.1062921021035729, -0.6027521208404681, -2.50337521912297, 0.13283492899586027, 2.5984230209111456, -1.443825125350671]
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

# with frame shift:
# f*p*q_r = q_m
# q_m*p_i*f_i = q_r
def find_rot_offset(q_r):
    fs_f = normalize(FRAME_CORR) # frame shift forward
    fs_i = qi(fs_f)
    q_ri = qi(q_r)
    p = qm(fs_i,qm(q_ri, normalize(BASE_ROT)))
    offset = qm(fs_f, p)
    restore = qi(offset)
    return normalize(offset), normalize(restore)

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector ** 2))


def msg_from_row_corrected(df):
    msgs = []
    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    offs, rot_restore = find_rot_offset(quat_from_pose(msg))
    pos_offset = np.asarray(point_from_pose(msg))
    i = 0
    for _, row in df.iterrows():
        #i+=1
        #if i == 128:
        #    break
        msgs.append(msg)
        msg = Pose()
        pose_from_point(msg, row[["position."+c for c in "xyz"]].values * 0.5 + pos_offset)
        pose_from_quat(msg, normalize(qm(row[["orientation."+c for c in "xyzw"]].values, rot_restore)))
    return msgs

def msg_from_model(model):
    msgs = []
    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    rot_offset, rot_restore = find_rot_offset(quat_from_pose(msg))
    pos_offset = np.asarray(point_from_pose(msg))
    output_state = np.asarray([0.0]*3 + list(qm(quat_from_pose(msg), init_rot)) + [0]).reshape(1,-1)
    t_base = np.asarray([0.01]).reshape(1,-1)
    for i in range(64):
        msgs.append(msg)
        t = t_base * i
        output_state = output_state.astype(np.float32)
        state = np.concatenate([t, output_state, np.asarray(TARGET_COORDS).reshape(1,-1)], axis=1)
        output_state = model(state).numpy().astype(np.float64)
        output_state[0,3:7] = normalize(output_state[0,3:7])
        msg = Pose()
        pose_from_point(msg, output_state[0,(0,1,2)] * 0.5 + pos_offset)
        #pose_from_quat(msg, init_rot)
        pose_from_quat(msg, qm(output_state[0,3:7], rot_restore))
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


def gripper_close(t: JointTrajectory):
    t.joint_names += ["panda_finger_joint1", "panda_finger_joint2"]
    t.points[0].positions = t.points[0].positions + (0.035, 0.035)
    for point in t.points[1:]:
        point.positions = point.positions + (0,0)

def execute_trajectory(df: pd.DataFrame):
    # for using a static dataframe
    #msgs = msg_from_row_corrected(df)
    # For using a policy model
    model = models.load_model(MODEL_FILE)
    msgs = msg_from_model(model)
    p,_ = group.compute_cartesian_path([x for x in msgs], 0.1, 0.0)
    p.joint_trajectory = rescale_time(p.joint_trajectory, 1)
    gripper_close(p.joint_trajectory)
    for pp in p.joint_trajectory.points:
        pp.velocities = []
        pp.accelerations = []
    group.execute(p, wait=True)

if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    #group_name = "manipulator"
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
    #joint_goal = group.get_current_joint_values()
    #joint_goal[0] = np.pi / 2
    #joint_goal[6] = np.pi / -4
    group.go(JOINT_GOAL, wait=True)
    group.stop()
    execute_trajectory(df)