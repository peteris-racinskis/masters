#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows where
from ntpath import join
from platform import release
from black import out
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
import pandas as pd
import numpy as np
import copy
import sys
import moveit_commander
from tf.transformations import quaternion_inverse as qi, quaternion_multiply as qm
from typing import List
import tensorflow
from tensorflow.keras import models

SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="world"
EE_LINK="panda_link8"
IFILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal-0-first-attempt.csv"
IFILE="/home/user/repos/masters/processed_data/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"
MODEL_FILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal"
#MODEL_FILE="/home/user/repos/masters/models/BCO-256x2-256x2-start-timesignal-doubled-noreg-ep200-b64-norm-gen"
#MODEL_FILE="/home/user/repos/masters/models/naiveBCx2048x2-ep20-norm-start-timesignal"
#MODEL_FILE="/home/user/repos/masters/models/naiveBCx128x2-ep20-norm-start-timesignal-quatreg"
MODEL_FILE="/home/user/repos/masters/models/naiveBCx128x2-ep100-norm-start-timesignal-partloss"
BASE_ROT=np.asarray([0.023,-0.685,0.002,-0.729])
BASE_ROT=np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973])

#BASE_ROT=np.asarray([ 0.46312436,  0.51316392, -0.48667049,  0.52832292])
JOINT_GOAL=[1.577647875986087, 0.1062921021035729, -0.6027521208404681, -2.50337521912297, 0.13283492899586027, 2.5984230209111456, -1.443825125350671]
#JOINT_GOAL=[0, 0.1062921021035729, -0.6027521208404681, -2.50337521912297, 0.13283492899586027, 2.5984230209111456, -1.443825125350671]
FRAME_CORR=np.asarray([0,-1,0,1])
#TARGET_COORDS=np.asarray([-3.6,-0.178823692236341,-0.36553905703157])
TARGET_COORDS=np.asarray([-2.5049639 , -0.03949555, -0.30162135])

'''
    WHAT I WAS DOING WRONG:
    1. The rotation frame correction was not actually being applied to the initial state (I was 
    multiplying by init_rot, which is the initial quaternion itself!);
    2. The starting coordinates were out of distribution for the model in question
'''



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
    #p = qm(qm(normalize(BASE_ROT), q_ri), fs_i)
    offset = qm(fs_f, p)
    restore = qi(offset)
    return normalize(offset), normalize(restore)
    #return normalize(fs_f), normalize(fs_i)

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector ** 2))

def release_threshold(model_output):
    return 0.035 if model_output >= 0.95 else 0

def release_time_fraction(states: List):
    for s in states:
        if s != 0:
            return states.index(s) / len(states)
    return 1

def msg_from_row_corrected(df):
    msgs = []
    released = []
    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    offs, rot_restore = find_rot_offset(quat_from_pose(msg))
    pos_offset = np.asarray(point_from_pose(msg))
    i = 0
    for _, row in df.iterrows():
        i+=1
        if i == 79:
            break
        msgs.append(msg)
        msg = Pose()
        pose_from_point(msg, row[["position."+c for c in "xyz"]].values * 0.5 + pos_offset)
        pose_from_quat(msg, normalize(qm(row[["orientation."+c for c in "xyzw"]].values, rot_restore)))
        released.append(release_threshold(row["Released"]))
    return msgs, released

def msg_from_model(model):
    msgs = []
    uncorr_msgs = []
    released = []
    msg = group.get_current_pose().pose
    u_msg = copy.deepcopy(msg)
    init_rot = quat_from_pose(msg)
    rot_offset, rot_restore = find_rot_offset(quat_from_pose(msg))
    pos_offset = np.asarray(point_from_pose(msg))
    offs_target = TARGET_COORDS + pos_offset
    output_state = np.asarray([0.0]*3 + list(qm(quat_from_pose(msg), rot_offset)) + [0]).reshape(1,-1) # was rotating the initial point by itself, not the computed offset
    pose_from_quat(u_msg, output_state[0,3:7])
    t_base = np.asarray([0.01]).reshape(1,-1)
    for i in range(64):
        msgs.append(msg)
        uncorr_msgs.append(u_msg)
        t = t_base * i
        output_state = output_state.astype(np.float32)
        state = np.concatenate([t, output_state, np.asarray(TARGET_COORDS).reshape(1,-1)], axis=1)
        output_state = model(state).numpy().astype(np.float64)
        #output_state[0,3:7] = normalize(output_state[0,3:7])
        msg = Pose()
        u_msg = Pose()
        pose_from_point(u_msg, output_state[0,(0,1,2)] * 1.0 + pos_offset)
        pose_from_point(msg, output_state[0,(0,1,2)] * 1.0 + pos_offset)
        #pose_from_quat(msg, init_rot)
        pose_from_quat(u_msg, normalize(output_state[0,3:7]))
        pose_from_quat(msg, qm(normalize(output_state[0,3:7]), rot_restore))
        released.append(release_threshold(output_state[0,-1]))
    return msgs, uncorr_msgs, released, offs_target

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

def gripper_close(t: JointTrajectory, release_fraction):
    t.joint_names += ["panda_finger_joint1", "panda_finger_joint2"]
    t_end = t.points[-1].time_from_start
    finish_time = int_time_to_float(t_end.secs, t_end.nsecs)
    release_time = release_fraction * finish_time
    gripper_start = robot.get_current_state().joint_state.position[-2:]
    #t.points[0].positions = t.points[0].positions + gripper_start
    for point in t.points[:]:
        t_point = point.time_from_start
        finger_state = (0.035,0.035) if int_time_to_float(t_point.secs, t_point.nsecs) > release_time else (0,0)
        point.positions = point.positions + finger_state

def msgs_to_csv(msgs: List[Pose], released: List, offs_target):
    fname = "testpackage/generated_trajectory_points.csv"
    df = pd.DataFrame({
        "Time": [0 for p in msgs],
        "x": [p.position.x for p in msgs],
        "y": [p.position.y for p in msgs],
        "z": [p.position.z for p in msgs],
        "rx": [p.orientation.x for p in msgs],
        "ry": [p.orientation.y for p in msgs],
        "rz": [p.orientation.z for p in msgs],
        "rw": [p.orientation.w for p in msgs],
        "Released": [float(r > 0) for r in released + released],
        "x-t": [offs_target[0] for p in msgs],
        "y-t": [offs_target[1] for p in msgs],
        "z-t": [offs_target[2] for p in msgs],
        "quaternion_norm": [0 for p in msgs],
    })
    df.to_csv(fname, index=False)

def execute_trajectory(df: pd.DataFrame):
    # for using a static dataframe
    #msgs, released = msg_from_row_corrected(df)
    offs_target = np.zeros(3)
    # For using a policy model
    # For loading the model with a custom loss
    custom_objects = {"quaternion_normalized_huber_loss": None}
    with tensorflow.keras.utils.custom_object_scope(custom_objects):
        model = models.load_model(MODEL_FILE)
    model = models.load_model(MODEL_FILE)
    msgs, u_msgs, released, offs_target = msg_from_model(model)
    msgs_to_csv(msgs + u_msgs, released, offs_target)
    release_fraction = release_time_fraction(released)
    p,_ = group.compute_cartesian_path([x for x in msgs], 0.01, 0.0)
    p.joint_trajectory = rescale_time(p.joint_trajectory, 20)
    gripper_close(p.joint_trajectory, release_fraction)
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
    joint_goal = group.get_current_joint_values()
    #joint_goal[0] = np.pi / 2
    #joint_goal[6] = np.pi / -4
    jnames = robot.get_current_state().joint_state.name
    current = JointTrajectoryPoint()
    current.positions = robot.get_current_state().joint_state.position
    target = JointTrajectoryPoint()
    target.positions = tuple(JOINT_GOAL + [0.00,0.00])
    trajectory = JointTrajectory()
    trajectory.points.append(current)
    trajectory.points.append(target)
    trajectory.joint_names = jnames
    rtr = RobotTrajectory()
    rtr.joint_trajectory = trajectory

    #group.go(JOINT_GOAL, wait=True)
    group.execute(rtr, wait=True)
    group.stop()
    execute_trajectory(df)