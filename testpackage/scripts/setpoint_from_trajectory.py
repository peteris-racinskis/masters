#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows 
from time import sleep
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryActionGoal, FollowJointTrajectoryAction, GripperCommandAction, GripperCommandActionGoal
import pandas as pd
import numpy as np
import copy
import sys
import moveit_commander
import actionlib
from tf.transformations import quaternion_inverse as qi, quaternion_multiply as qm
from typing import List
import tensorflow
from tensorflow.keras import models

SETPOINT_CHANNEL="/pid_setpoint"
WORLDLINK="world"
EE_LINK="panda_link8"
#IFILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal-0-first-attempt.csv"
#IFILE="/home/user/repos/masters/processed_data_old/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"
IFILE="/home/user/repos/masters/processed_data/train_datasets/train-start-time-doubled-7db3d40f19abc9f24f46.csv"
MODEL_FILE="/home/user/repos/masters/models/naiveBC-norm-start-timesignal"
#MODEL_FILE="/home/user/repos/masters/models/BCO-256x2-256x2-start-timesignal-doubled-noreg-ep200-b64-norm-gen"
#MODEL_FILE="/home/user/repos/masters/models/naiveBCx2048x2-ep20-norm-start-timesignal"
#MODEL_FILE="/home/user/repos/masters/models/naiveBCx128x2-ep20-norm-start-timesignal-quatreg"
MODEL_FILE="/home/user/repos/masters/models/naiveBCx128x2-newdata-ep100-norm-start-timesignal-partloss"
#MODEL_FILE="/home/user/repos/masters/models/naiveBCx128x2-ep100-norm-start-timesignal-partloss"
#BASE_ROT=np.asarray([0.023,-0.685,0.002,-0.729])
#BASE_ROT=np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973])
BASE_ROT=np.asarray([-0.700575220584869,0.006480328785255,0.712678784132004,0.03341787615791])
#BASE_ROT=np.asarray([-0.595393347740173,-0.037378676235676, 0.794532498717308,0.04247132204473])
#    init_orientation = np.asarray([-0.595393347740173,-0.037378676235676, 0.794532498717308,0.04247132204473])

#BASE_ROT=np.asarray([ 0.46312436,  0.51316392, -0.48667049,  0.52832292])
#JOINT_GOAL=[1.577647875986087, 0.1062921021035729, -0.6027521208404681, -2.50337521912297, 0.13283492899586027, 2.5984230209111456, -1.443825125350671]
#JOINT_GOAL=[1.3963414368265257, -1.673500445079532, 2.0627445115287806, -2.0407557772110856, -1.5704981923339751, 1.38]
#JOINT_GOAL=[-2.140, -1.621, -2.010, -1.031, 1.558, 1.532]
JOINT_GOAL=[-2.267834011708395, -1.8479305706419886, -1.8413517475128174, -1.0335948032191773, 1.5439883470535278, 2.3857996463775635]
JOINT_GOAL=[-1.2, -1.8479305706419886, -1.8413517475128174, -1.0335948032191773, 1.5439883470535278, 2.3857996463775635]
#JOINT_GOAL=[0, 0.1062921021035729, -0.6027521208404681, -2.50337521912297, 0.13283492899586027, 2.5984230209111456, -1.443825125350671]
FRAME_CORR=np.asarray([0,-1,0,1])
#TARGET_COORDS=np.asarray([-3.6,-0.178823692236341,-0.36553905703157])
TARGET_COORDS=np.asarray([-1.8 , 0.05, -0.4162135])
SCALER=1
RELEASE_THRESH=0.95
TOOL_OFFSET=np.asarray([0.0,0.0,-0.05])
#EE_LINK="ee_link"
EE_LINK="tool0"

'''
    WHAT I WAS DOING WRONG:
    1. The rotation frame correction was not actually being applied to the initial state (I was 
    multiplying by init_rot, which is the initial quaternion itself!);
    2. The starting coordinates were out of distribution for the model in question
'''



def point_from_pose(pose: Pose, numpy=False) -> List:
    p = pose.position
    point = [p.x, p.y, p.z]
    return np.asarray(point) if numpy else point

def quat_from_pose(pose: Pose, numpy=False) -> List:
    o = pose.orientation
    quat = [o.x, o.y, o.z, o.w]
    return np.asarray(quat) if numpy else quat

def pose_from_point(pose: Pose, point) -> Pose:
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]

def pose_from_quat(pose: Pose, quat) -> Pose:
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

def apply_rotation(q, v):
    q_i = qi(q)
    v_q = np.concatenate([v, np.zeros(1)])
    return qm(qm(q, v_q), q_i)[:-1]

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
    return 0.035 if model_output >= RELEASE_THRESH else 0

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
        if i < 64:
            continue
        if i == 107:
            break
        msgs.append(msg)
        msg = Pose()
        pose_from_point(msg, row[["position."+c for c in "xyz"]].values * SCALER + pos_offset)
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
    #for i in range(64):
    for i in range(100):
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
        pose_from_point(msg, output_state[0,(0,1,2)] * SCALER + pos_offset)
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

def gripper_close_ur5(t: JointTrajectory, release_fraction):
    t_g = copy.deepcopy(t)
    t_end = t_g.points[-1].time_from_start
    t_g.joint_names = ["finger_joint"]
    finish_time = int_time_to_float(t_end.secs, t_end.nsecs)
    release_time = release_fraction * finish_time
    for point in t_g.points:
        t_point = point.time_from_start
        finger_state = (0.02,0.02) if int_time_to_float(t_point.secs, t_point.nsecs) > release_time else (0.78,0.78)
        point.positions = finger_state
    gtr = RobotTrajectory()
    gtr.joint_trajectory = t_g
    return gtr


def gripper_close(t: JointTrajectory, release_fraction):
    #t.joint_names += ["panda_finger_joint1", "panda_finger_joint2"]
    t.joint_names += ["finger_joint"]
    t_end = t.points[-1].time_from_start
    finish_time = int_time_to_float(t_end.secs, t_end.nsecs)
    release_time = release_fraction * finish_time
    gripper_start = robot.get_current_state().joint_state.position[-2:]
    #t.points[0].positions = t.points[0].positions + gripper_start
    for point in t.points[:]:
        t_point = point.time_from_start
        #finger_state = (0.035,0.035) if int_time_to_float(t_point.secs, t_point.nsecs) > release_time else (0,0)
        finger_state = (0.02,0.02) if int_time_to_float(t_point.secs, t_point.nsecs) > release_time else (0.78,0.78)
        point.positions = point.positions + finger_state

def msgs_to_csv(msgs: List[Pose], released: List, offs_target):
    fname = f"testpackage/generated_trajectory_points-{SCALER}.csv"
    df = pd.DataFrame({
        "Time": [0 for p in msgs],
        "x": [p.position.x for p in msgs],
        "y": [p.position.y for p in msgs],
        "z": [p.position.z for p in msgs],
        "rx": [p.orientation.x for p in msgs],
        "ry": [p.orientation.y for p in msgs],
        "rz": [p.orientation.z for p in msgs],
        "rw": [p.orientation.w for p in msgs],
        "Released": [float(r > 0) for r in released + released + released],
        "x-t": [offs_target[0] for p in msgs],
        "y-t": [offs_target[1] for p in msgs],
        "z-t": [offs_target[2] for p in msgs],
        "quaternion_norm": [0 for p in msgs],
    })
    df.to_csv(fname, index=False)

def follow_joint_action(trajectory):
    goal = FollowJointTrajectoryActionGoal()
    goal.header.stamp = rospy.Time.now()
    goal.goal_id.stamp = rospy.Time.now()
    goal.goal.trajectory = trajectory
    goal.goal.path_tolerance = []
    goal.goal.goal_tolerance = []
    client.send_goal(goal.goal)

def gripper_action(target=1.0):
    g_goal = GripperCommandActionGoal()
    g_goal.header.stamp = rospy.Time.now()
    g_goal.goal_id.stamp = rospy.Time.now()
    g_goal.goal.command.position = target
    g_client.send_goal(g_goal.goal)


    '''

    PATH TO TRAJECTORY OPTIMIZER CODE:
    /home/ur5e-robopc/ros_workspace/src/VIZTA_robot_control/vizta/src/trajectory_generation.cpp
    

    '''


def shift_trajectory(msgs: List[Pose], shift=TOOL_OFFSET):
    offset_msgs = [copy.deepcopy(msgs[0])]
    # Get the shift of the first point. It will remain at the origin so no need to shifts
    start_rot = quat_from_pose(msgs[0], True)
    first_shift = apply_rotation(start_rot, shift)
    # Get the shift of all the other points, apply, then subtract the first shift
    for point in msgs[1:]:
        pos, rot = point_from_pose(point, True), quat_from_pose(point, True)
        point_shift = apply_rotation(rot, shift)
        new_pos = pos + point_shift - first_shift
        offset_msg = Pose()
        pose_from_point(offset_msg, new_pos)
        pose_from_quat(offset_msg, rot)
        offset_msgs.append(offset_msg)
    return offset_msgs

def execute_trajectory(df: pd.DataFrame):
    # for using a static dataframe
    msgs, released = msg_from_row_corrected(df)
    offs_target = np.zeros(3)
    # For using a policy model
    # For loading the model with a custom loss
    custom_objects = {"quaternion_normalized_huber_loss": None}
    #with tensorflow.keras.utils.custom_object_scope(custom_objects):
    #    model = models.load_model(MODEL_FILE)
    #msgs, u_msgs, released, offs_target = msg_from_model(model)
    o_msgs = shift_trajectory(msgs)
    msgs_to_csv(msgs + o_msgs + msgs, released, offs_target)
    #msgs_to_csv(msgs + o_msgs + u_msgs, released, offs_target)
    release_fraction = release_time_fraction(released)
    p, f = group.compute_cartesian_path([x for x in o_msgs], 0.05, 0.0)
    #p, f = group.compute_cartesian_path([x for x in msgs], 0.05, 0.0)
    totaltime = 20
    opentime = release_fraction * totaltime
    p.joint_trajectory = rescale_time(p.joint_trajectory, totaltime)
    p.joint_trajectory.points[1].time_from_start.nsecs += 200
    for pp in p.joint_trajectory.points:
        pp.velocities = []
        pp.accelerations = []
    #gripper_close(p.joint_trajectory, release_fraction)
    #g = gripper_close_ur5(p.joint_trajectory, release_fraction)
    pass
    follow_joint_action(p.joint_trajectory)
    sleep(opentime)
    gripper_action(0.0)
    #group.execute(p, wait=False)
    #gripper_group.execute(g, wait=False)
    pass

if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    #group_name = "panda_arm"
    group_name = "arm"
    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_end_effector_link("tool0")
    gripper_group = moveit_commander.MoveGroupCommander("gripper")
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
    #current.positions = robot.get_current_state().joint_state.position[:7]
    current.positions = robot.get_current_state().joint_state.position[:6]
    target = JointTrajectoryPoint()
    target.positions = tuple(JOINT_GOAL)
    #target.positions = tuple(JOINT_GOAL + [0.78])

    target.time_from_start.secs = 1
    trajectory = JointTrajectory()
    trajectory.points.append(copy.deepcopy(current))
    trajectory.points.append(target)
    #trajectory.joint_names = jnames[:7]
    trajectory.joint_names = jnames[:6]
    rtr = RobotTrajectory()
    rtr.joint_trajectory = trajectory

    '''
    grip_target = JointTrajectoryPoint()
    grip_target.positions = tuple([0.78])
    grip_target.time_from_start.secs = 1
    #grip_trajectory = JointTrajectory()
    grip_trajectory.joint_names = (jnames[6],)
    current.positions = (current.positions[6],)
    grip_trajectory.points.append(current)
    grip_trajectory.points.append(grip_target)
    gtr = RobotTrajectory()
    gtr.joint_trajectory = grip_trajectory
    '''

    #group.go(JOINT_GOAL, wait=True)

    goal = FollowJointTrajectoryActionGoal()
    goal.header.stamp = rospy.Time.now()
    goal.goal_id.stamp = rospy.Time.now()
    goal.goal.trajectory = trajectory
    goal.goal.path_tolerance = []
    goal.goal.goal_tolerance = []
    client = actionlib.SimpleActionClient("/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    client.wait_for_server()
    client.send_goal(goal.goal)
    #topic = "/arm_controller/follow_joint_trajectory/goal"
    #pub = rospy.Publisher(topic,FollowJointTrajectoryActionGoal,queue_size=10)
    #rospy.init_node("talker")
    #pub.publish(goal)


    g_goal = GripperCommandActionGoal()
    g_goal.header.stamp = rospy.Time.now()
    g_goal.goal_id.stamp = rospy.Time.now()
    g_goal.goal.command.position = 1.0
    g_client = actionlib.SimpleActionClient("/gripper/gripper_cmd", GripperCommandAction)
    g_client.wait_for_server()
    g_client.send_goal(g_goal.goal)

    #group.execute(rtr, wait=True)
    #gripper_group.execute(gtr, wait=True)
    group.stop()
    gripper_group.stop()
    execute_trajectory(df)