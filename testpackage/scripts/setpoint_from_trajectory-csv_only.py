#!/usr/bin/env python3
# shebang needed because this script gets called from somewhere who knows 
from time import sleep
import rospy
from geometry_msgs.msg import Pose
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

IFILE="testpackage/generated_trajectory_points_.csv"
BASE_ROT=np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973])
JOINT_GOAL=[1.3963414368265257, -1.673500445079532, 2.0627445115287806, -2.0407557772110856, -1.5704981923339751, 1.38]
FRAME_CORR=np.asarray([0,0,0,1])
TARGET_COORDS=np.asarray([-2.6 , 0.05, -0.30162135])
GROUP_NAME = "arm"
GRIPPER_NAME = "gripper"

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
        #pose_from_point(msg, row[["position."+c for c in "xyz"]].values * 0.5 + pos_offset)
        pose_from_point(msg, row[[c for c in "xyz"]].values * 0.3 + pos_offset)
        #pose_from_quat(msg, normalize(qm(row[["orientation."+c for c in "xyzw"]].values, rot_restore)))
        pose_from_quat(msg, normalize(qm(row[["r"+c for c in "xyzw"]].values, rot_restore)))
        released.append(release_threshold(row["Released"]))
    return msgs, released

def int_time_to_float(secs, nsecs):
    return secs + 1e-9 * nsecs

def float_time_to_int(time):
    secs = int(time)
    nsecs = int(1e9*(time-secs))
    return secs, nsecsrosr

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
    t.joint_names += ["finger_joint"]
    t_end = t.points[-1].time_from_start
    finish_time = int_time_to_float(t_end.secs, t_end.nsecs)
    release_time = release_fraction * finish_time
    for point in t.points[:]:
        t_point = point.time_from_start
        finger_state = (0.02,0.02) if int_time_to_float(t_point.secs, t_point.nsecs) > release_time else (0.78,0.78)
        point.positions = point.positions + finger_state

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

def execute_trajectory(df: pd.DataFrame):
    msgs, released = msg_from_row_corrected(df)
    release_fraction = release_time_fraction(released)
    p, f = group.compute_cartesian_path([x for x in msgs], 0.05, 0.0)
    totaltime = 0.64
    opentime = release_fraction * 0.5 * totaltime
    p.joint_trajectory = rescale_time(p.joint_trajectory, totaltime)
    p.joint_trajectory.points[1].time_from_start.nsecs += 200
    for pp in p.joint_trajectory.points:
        pp.velocities = []
    follow_joint_action(p.joint_trajectory)
    sleep(opentime)
    gripper_action(0.0)

if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(GROUP_NAME)
    gripper_group = moveit_commander.MoveGroupCommander(GRIPPER_NAME)
    planning_frame = group.get_planning_frame()
    print(f"============ Reference frame: {planning_frame}")
    eef_link = group.get_end_effector_link()
    print(f"============ End effector link: {eef_link}")
    group_names = robot.get_group_names()
    print(f"============ Available Planning Groups: {robot.get_group_names()}")


    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")
    joint_goal = group.get_current_joint_values()
    jnames = robot.get_current_state().joint_state.name
    current = JointTrajectoryPoint()
    current.positions = robot.get_current_state().joint_state.position[:6]
    target = JointTrajectoryPoint()
    target.positions = tuple(JOINT_GOAL)

    target.time_from_start.secs = 1
    trajectory = JointTrajectory()
    trajectory.points.append(copy.deepcopy(current))
    trajectory.points.append(target)
    trajectory.joint_names = jnames[:6]
    rtr = RobotTrajectory()
    rtr.joint_trajectory = trajectory
    
    client = actionlib.SimpleActionClient("/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    client.wait_for_server()
    follow_joint_action(trajectory)

    g_client = actionlib.SimpleActionClient("/gripper/gripper_cmd", GripperCommandAction)
    g_client.wait_for_server()
    gripper_action()

    sleep(3)
    group.stop()
    gripper_group.stop()
    execute_trajectory(df)