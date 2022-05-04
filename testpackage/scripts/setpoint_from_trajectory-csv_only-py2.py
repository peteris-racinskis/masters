#!/usr/bin/env python
# shebang needed because this script gets called from somewhere who knows 
from time import sleep
import rospy
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output
from control_msgs.msg import FollowJointTrajectoryActionGoal, FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, GripperCommandAction, GripperCommandActionGoal
import pandas as pd
import numpy as np
import copy
import sys
import moveit_commander
import actionlib
from tf.transformations import quaternion_inverse as qi, quaternion_multiply as qm
from tf import TransformListener

import pdb

WORLDLINK="world_frame"
BASE_LINK="base_link"
EE_LINK="tool0"

IFILE="generated_trajectory_points-0.3.csv"
FOLLOW_JOINT_TRAJECTORY_TOPIC="/scaled_pos_joint_traj_controller/follow_joint_trajectory"
BASE_ROT=np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973])
JOINT_GOAL=[1.3963414368265257, -1.673500445079532, 2.0627445115287806, -2.0407557772110856, -1.5704981923339751, 1.38]
FRAME_CORR=np.asarray([0,0,0,1])
TARGET_COORDS=np.asarray([-2.6 , 0.05, -0.30162135])
GROUP_NAME = "manipulator"
GRIPPER_NAME = "gripper"
PLANNING_STEP = 0.1
PLANNING_JUMP_LIM = 0.0
TOTALTIME=20


class FollowTrajectoryWrapper():

    def __init__(self, gripper_action_wrapper):
        self._client = actionlib.SimpleActionClient(FOLLOW_JOINT_TRAJECTORY_TOPIC, FollowJointTrajectoryAction)
        self._client.wait_for_server()
        self._gripper_action_wrapper = gripper_action_wrapper

    def throwing_motion(self, trajectory):
        goal = FollowJointTrajectoryActionGoal()
        goal.header.stamp = rospy.Time.now()
        goal.goal_id.stamp = rospy.Time.now()
        goal.goal.trajectory = trajectory
        goal.goal.path_tolerance = []
        goal.goal.goal_tolerance = []
        self._released = False
        self._client.send_goal(goal.goal, feedback_cb=self._callback)
        self._client.wait_for_result(rospy.Duration(TOTALTIME))

    def _callback(self, feedback):
        desired = feedback.desired.positions
        actual = feedback.actual.positions
        if not self._released and (np.allclose(desired, self._objective, rtol=1e-2) or np.allclose(actual, self._objective, rtol=1e-2)):
            self._released = True
            self._gripper_action_wrapper.gripper_open()
            self._log("release: {}".format(self._objective))
            self._log("desired: {}".format(desired))
            self._log("actual: {}".format(actual))

    def _log(self, msg):
        rospy.loginfo("{}".format(msg))

    def set_callback_objective(self, joint_target):
        self._objective = joint_target

class GripperActionWrapper():

    def __init__(self):
        self._pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output, queue_size=1)
        self._msg = Robotiq2FGripper_robot_output()

    def _msg_reset(self):
        self._msg.rACT = 0 # activation
        self._msg.rGTO = 0 # go to objective
        self._msg.rATR = 0 # emergency release
        self._msg.rPR = 0 # position request open (0) / closed (255)
        self._msg.rSP = 255 # speed
        self._msg.rFR = 127 # force


    def _gripper_action(self):
        self._pub.publish(self._msg)
        self._msg_reset()
        rospy.loginfo("Gripper action was triggered!!")

    def _gripper_set_state(self, state):
        self._msg.rACT = state
        self._gripper_action()

    def gripper_activate(self):
        self._gripper_set_state(1)

    def gripper_deactivate(self):
        self._gripper_set_state(0)

    def _gripper_goto(self):
        self._msg.rGTO = 1
        self._gripper_action()

    def gripper_close(self):
        self._msg.rPR = 255
        self._gripper_goto()

    def gripper_open(self):
        self._msg.rPR = 0
        self._gripper_goto()


def point_from_pose(pose):
    p = pose.position
    return [p.x, p.y, p.z]

def quat_from_pose(pose):
    o = pose.orientation
    return [o.x, o.y, o.z, o.w]

def pose_from_point(pose, point):
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]

def pose_from_quat(pose, quat):
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

def normalize(vector):
    return vector / np.sqrt(np.sum(vector ** 2))

def release_threshold(model_output):
    return 0.035 if model_output >= 0.95 else 0

def release_time_fraction(states):
    for s in states:
        if s != 0:
            return states.index(s) / len(states), states.index(s)
    return 1

def apply_rotation(q, v):
    q_i = qi(q)
    v_q = np.concatenate([v, np.zeros(1)])
    return qm(qm(q, v_q), q_i)[:-1]

def msgs_to_csv(msgs, released, offs_target):
    fname = "transformed_trajectory_points.csv"
    columns=["Time", "x", "y", "z", "rx", "ry", "rz", "rw", "Released", "x-t", "y-t", "z-t", "quaternion_norm"]
    df = pd.DataFrame(data={
        "Time": [0 for p in msgs],
        "x": [p.position.x for p in msgs],
        "y": [p.position.y for p in msgs],
        "z": [p.position.z for p in msgs],
        "rx": [p.orientation.x for p in msgs],
        "ry": [p.orientation.y for p in msgs],
        "rz": [p.orientation.z for p in msgs],
        "rw": [p.orientation.w for p in msgs],
        "Released": [float(r > 0) for r in released],
        "x-t": [offs_target[0] for p in msgs],
        "y-t": [offs_target[1] for p in msgs],
        "z-t": [offs_target[2] for p in msgs],
        "quaternion_norm": [0 for p in msgs],
    }, columns=columns)
    #pdb.set_trace()
    df.to_csv(fname, index=False)

def msg_from_row_corrected(df):
    msgs = []
    released = []
    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    init_pos = np.asarray(point_from_pose(msg))
    offs, rot_restore = find_rot_offset(init_rot)
    i = 0
    for _, row in df.iterrows():
        if i == 0:
            pos_offset = init_pos - row[[c for c in "xyz"]].values
            offs_target = TARGET_COORDS + pos_offset
        i+=1
        if i == 64:
            break
        msgs.append(msg)
        msg = Pose()
        pose_from_point(msg, row[[c for c in "xyz"]].values + pos_offset)
        pose_from_quat(msg, normalize(qm(row[["r"+c for c in "xyzw"]].values, rot_restore)))
        released.append(release_threshold(row["Released"]))
    return msgs, released, offs_target

def int_time_to_float(secs, nsecs):
    return secs + 1e-9 * nsecs

def float_time_to_int(time):
    secs = int(time)
    nsecs = int(1e9*(time-secs))
    return secs, nsecs

def rescale_time(trajectory, dt=1.0):
    finish_time_int = trajectory.points[-1].time_from_start
    finish_time = int_time_to_float(finish_time_int.secs, finish_time_int.nsecs)
    scaler = dt / finish_time
    for point in trajectory.points:
        newtime_int = point.time_from_start
        newtime = scaler * int_time_to_float(newtime_int.secs, newtime_int.nsecs)
        point.time_from_start.secs, point.time_from_start.nsecs = float_time_to_int(newtime)
    return trajectory

def gripper_action(target=1.0):
    g_goal = GripperCommandActionGoal()
    g_goal.header.stamp = rospy.Time.now()
    g_goal.goal_id.stamp = rospy.Time.now()
    g_goal.goal.command.position = target
    g_client.send_goal(g_goal.goal)

def execute_trajectory(df):
    msgs, released, offs_target = msg_from_row_corrected(df)
    msgs_to_csv(msgs, released, offs_target)
    release_fraction, release_index = release_time_fraction(released)

    p_release, _ = group.compute_cartesian_path(msgs[:release_index+1], PLANNING_STEP, PLANNING_JUMP_LIM)
    joint_target_release = p_release.joint_trajectory.points[-1].positions
    p, f = group.compute_cartesian_path([x for x in msgs], PLANNING_STEP, PLANNING_JUMP_LIM)
    pdb.set_trace()
    p = group.retime_trajectory(group.get_current_state(), p, 1.0, algorithm="time_optimal_trajectory_generation")
    pdb.set_trace()
    opentime = release_fraction * TOTALTIME
    for pp in p.joint_trajectory.points[1:]:
        pp.time_from_start.nsecs += 5e8
    p.joint_trajectory = rescale_time(p.joint_trajectory, TOTALTIME)
    for pp in p.joint_trajectory.points:
        pp.velocities = []
        pp.accelerations = []

    follow_trajectory_wrapper.set_callback_objective(joint_target_release)
    pdb.set_trace()
    follow_trajectory_wrapper.throwing_motion(p.joint_trajectory)



if __name__ == "__main__":
    df = pd.read_csv(IFILE)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(GROUP_NAME)

    group.set_end_effector_link("tool0")
    jnames = robot.get_current_state().joint_state.name
    current = JointTrajectoryPoint()
    current.positions = robot.get_current_state().joint_state.position[:6]
    target = JointTrajectoryPoint()
    target.positions = tuple(JOINT_GOAL)

    target.time_from_start.secs = 10
    trajectory = JointTrajectory()
    trajectory.points.append(copy.deepcopy(current))
    trajectory.points.append(target)
    trajectory.joint_names = jnames[:6]
    rtr = RobotTrajectory()
    rtr.joint_trajectory = trajectory
    
    gripper_wrapper = GripperActionWrapper()
    follow_trajectory_wrapper = FollowTrajectoryWrapper(gripper_wrapper)

    group.execute(rtr, wait=True)
    group.stop()
    pdb.set_trace()
    gripper_wrapper.gripper_activate()
    gripper_wrapper.gripper_close()

    pdb.set_trace()
    execute_trajectory(df)