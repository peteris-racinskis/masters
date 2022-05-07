#!/usr/bin/env python
from time import sleep, time
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotTrajectory, RobotState
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
from ur_msgs.srv import SetIO

import pdb

WORLDLINK="world_frame"
BASE_LINK="base_link"
EE_LINK="tool0"

IFILE="generated_trajectory_points-1-5cm.csv"
FOLLOW_JOINT_TRAJECTORY_TOPIC="/scaled_pos_joint_traj_controller/follow_joint_trajectory"
SERVICE_PATH="/ur_hardware_interface/set_io"
FK_PATH="/compute_fk"
#BASE_ROT=np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973])
BASE_ROT=np.asarray([-0.700575220584869,0.006480328785255,0.712678784132004,0.03341787615791])
#JOINT_GOAL=[1.3963414368265257, -1.673500445079532, 2.0627445115287806, -2.0407557772110856, -1.5704981923339751, 1.38]
#JOINT_GOAL=[-2.267834011708395, -1.8479305706419886, -1.8413517475128174, -1.0335948032191773, 1.5439883470535278, 2.3857996463775635]
#JOINT_GOAL=[-1.2, -1.8479305706419886, -1.8413517475128174, -1.0335948032191773, 1.5439883470535278, 2.3857996463775635]
JOINT_GOAL=[-1.2, -1.8479305706419886, -1.8413517475128174, -1.0335948032191773, 1.5439883470535278, 2.78]
FRAME_CORR=np.asarray([0,0,0,1])
#FRAME_CORR=np.asarray([0,-1,0,1])
TARGET_COORDS=np.asarray([-2.6 , 0.05, -0.30162135])
GROUP_NAME = "manipulator"
GRIPPER_NAME = "gripper"
PLANNING_STEP = 0.2
PLANNING_JUMP_LIM = 0.0
TOTALTIME=0.8
START_MOVE_TIME=5
DOWNSAMPLING=16
OPT_ACCEL_SCALE=0.1
OPT_SPEED_SCALE=1
OPT_ALGORITHM="time_optimal_trajectory_generation"
#DF_START_INDEX=44
#DF_STOP_INDEX=87
DF_START_INDEX=100
DF_STOP_INDEX=160
RELEASE_ANGLE_TOLERANCE=5e-2
LINEAR_SEGMENT_LENGTH=0.15


class FKWrapper():

    def __init__(self, joint_names, service_path=FK_PATH, link=EE_LINK):
        self._client = rospy.ServiceProxy(service_path, GetPositionFK)
        self._msg = RobotState()
        self._msg.joint_state.name = joint_names
        self._header = Header()
        self._link = [link]

    def get_fk(self, joint_state):
        self._msg.joint_state.position = joint_state
        return self._client(self._header, self._link, self._msg)

    def get_fk_pos(self, joint_state):
        pose = self.get_fk(joint_state).pose_stamped[0].pose
        return point_from_pose(pose, True)


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
        self.time = time()
        self._client.send_goal(goal.goal, feedback_cb=self._callback, done_cb=self._done)
        self._client.wait_for_result(rospy.Duration(TOTALTIME + 1))

    def _done(self, _, __):
        self._log("Done time: {}".format(time()-self.time))

    def _callback(self, feedback):
        desired = feedback.desired.positions
        actual = feedback.actual.positions
        if not self._released and (np.allclose(desired[0:1], self._objective[0:1], rtol=RELEASE_ANGLE_TOLERANCE) or np.allclose(actual[0:1], self._objective[0:1], rtol=RELEASE_ANGLE_TOLERANCE)):
            self._log("Release time: {}".format(time() - self.time))
            self._released = True
            self._gripper_action_wrapper.gripper_open()
            self._log("release: {}".format(self._objective))
            self._log("desired: {}".format(desired))
            self._log("actual: {}".format(actual))

    def _log(self, msg):
        rospy.loginfo("{}".format(msg))

    def set_callback_objective(self, joint_target):
        self._objective = joint_target


class GripperServiceWrapper():

    def __init__(self, srv_path=SERVICE_PATH):
        self._proxy = rospy.ServiceProxy(SERVICE_PATH, SetIO)

    def gripper_open(self):
        self._proxy(1,1,1)

    def gripper_close(self):
        self._proxy(1,1,0)


class GripperActionWrapper():

    def __init__(self):
        self._pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output, queue_size=1)
        self._msg = Robotiq2FGripper_robot_output()
        self._msg.rACT = 0 # activation

    def _msg_reset(self):
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


def point_from_pose(pose, numpy=False):
    p = pose.position
    point = [p.x, p.y, p.z]
    return np.asarray(point) if numpy else point

def quat_from_pose(pose, numpy=False):
    o = pose.orientation
    quat = [o.x, o.y, o.z, o.w]
    return np.asarray(quat) if numpy else quat

def pose_from_point(pose, point):
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]

def pose_from_quat(pose, quat):
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    
def find_rot_offset(q_r, base=BASE_ROT):
    fs_f = normalize(FRAME_CORR) # frame shift forward
    fs_i = qi(fs_f)
    q_ri = qi(q_r)
    p = qm(fs_i,qm(q_ri, normalize(base)))
    offset = qm(fs_f, p)
    restore = qi(offset)
    #return [0,0,0,1], [0,0,0,1]
    return normalize(offset), normalize(restore)

def normalize(vector):
    return vector / np.sqrt(np.sum(vector ** 2))

def release_threshold(model_output):
    return 0.035 if model_output >= 0.8 else 0

def release_time_fraction(states):
    for s in states:
        if s != 0:
            return states.index(s) / len(states), states.index(s)
    return 1, len(states) - 1

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

def resample_path(msgs, released, factor=DOWNSAMPLING):
    r_msg = []
    r_released = []
    for i in range(len(msgs[:-1])):
        if i % factor == 0:
            r_msg.append(msgs[i])
            r_released.append(released[i])
    return r_msg + [msgs[-1]], r_released + [released[-1]]

def msg_from_row_corrected(df):
    msgs = []
    released = []

    msg = group.get_current_pose().pose
    init_rot = quat_from_pose(msg)
    init_pos = np.asarray(point_from_pose(msg))

    i = 0
    for _, row in df.iterrows():
        if i == DF_START_INDEX:
            pos_offset = init_pos - row[[c for c in "xyz"]].values
            offs_target = TARGET_COORDS + pos_offset
            base = row[["r"+c for c in "xyzw"]].values
            offs, rot_restore = find_rot_offset(init_rot, base)
        i+=1
        if i <= DF_START_INDEX:
            continue;
        if i == DF_STOP_INDEX:
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

def append_linear_segment(msgs, released):
    last, slast = point_from_pose(msgs[-1], True), point_from_pose(msgs[-2], True)
    final_orientation = quat_from_pose(msgs[-1])
    direction = normalize(last - slast) * LINEAR_SEGMENT_LENGTH
    final_position = last + direction
    new_point = Pose()
    pose_from_point(new_point, final_position)
    pose_from_quat(new_point, final_orientation)
    return msgs + [new_point], released + [1.0]

def v_len(vector):
    return np.linalg.norm(vector)

def close(f1, f2, tol=1e-2):
    return abs(f1-f2) > tol

def find_nearest_time_parametrization(msgs, joint_trajectory):
    # Assume: all joint states lie very close to cartesian line segments.
    # Assume: joint states much sparser than cartesian points.
    # Thus, simply find the closest straight line segment to each fk
    # and interpolate the time of its two termina.
    # Find nearest point for every joint state
    pdb.set_trace()
    cart_pos = []
    j_points = []
    indices = []
    neighbor_indices = []
    relative_timestamps = []

    for i in range(1, len(msgs)):
        #cart_pos.append((point_from_pose(msgs[i]), i))
        cart_pos.append(point_from_pose(msgs[i], True))
    pdb.set_trace()

    for point in joint_trajectory.points:
        j_position = fk_wrapper.get_fk_pos(point.positions)
        index = 0
        for j in range(len(cart_pos)):
            if v_len(j_position - cart_pos[j]) < v_len(j_position - cart_pos[index]):
                index = j
        j_points.append(j_position)
        indices.append(index)
    pdb.set_trace()

     # find which neighbor is nearest
    for j_point, index in zip(j_points, indices):
        if index == 0:
            neighbor_indices.append(1)
        elif index == len(cart_pos)-1:
            neighbor_indices.append(index-1)
        else:
            d_prev = v_len(j_position - cart_pos[index-1])
            d_next = v_len(j_position - cart_pos[index+1])
            if d_next >= d_prev:
                neighbor_indices.append(index+1)
            else:
                neighbor_indices.append(index-1)
    pdb.set_trace()

    for j_point, index, n_index in zip(j_points, indices, neighbor_indices):
        sign = 1 if index >= n_index else -1
        # vector between joint point and cartesian point
        d = j_point - cart_pos[index]
        # vector between cartesian points
        c = cart_pos[n_index] - cart_pos[index]
        # projection, divided by the base again
        p = np.dot(d,c) / (v_len(c) ** 2)
        relative_timestamp = (float(index) / len(cart_pos)) + (p * sign) * 0.01
        relative_timestamps.append(relative_timestamp)
    pdb.set_trace()

    trajectory = copy.deepcopy(joint_trajectory)
    trajectory.points = [copy.deepcopy(joint_trajectory.points[0])]


    for i in range(1, len(relative_timestamps[:-1])):
        if not close(relative_timestamps[i], relative_timestamps[i-1]) and not close(relative_timestamps[i], relative_timestamps[i+1]):
            point = copy.deepcopy(joint_trajectory.points[i])
            point.time_from_start.secs, point.time_from_start.nsecs = float_time_to_int(relative_timestamps[i] * TOTALTIME)
            trajectory.points.append(point)

    trajectory.points.append(copy.deepcopy(joint_trajectory.points[-1]))
    trajectory.points[-1].time_from_start.secs, trajectory.points[-1].time_from_start.nsecs = float_time_to_int(TOTALTIME * 1.0)
    pdb.set_trace()

    return trajectory


def execute_trajectory(df):
    msgs, released, offs_target = msg_from_row_corrected(df)
    msgs, released = append_linear_segment(msgs, released)
    msgs_to_csv(msgs, released, offs_target)
    r_msgs, released = resample_path(msgs, released)
    #msgs, released = append_linear_segment(msgs, released)
    release_fraction, release_index = release_time_fraction(released)
    #release_index -= 2

    p_release, _ = group.compute_cartesian_path(r_msgs[:release_index+1], PLANNING_STEP, PLANNING_JUMP_LIM)
    joint_target_release = p_release.joint_trajectory.points[-1].positions
    p, f = group.compute_cartesian_path([x for x in r_msgs], PLANNING_STEP, PLANNING_JUMP_LIM)

    p = group.retime_trajectory(group.get_current_state(), p, OPT_SPEED_SCALE, OPT_ACCEL_SCALE, algorithm=OPT_ALGORITHM)

    for pp in p.joint_trajectory.points[1:]:
        pp.time_from_start.nsecs += 5e8

    p.joint_trajectory = rescale_time(p.joint_trajectory, TOTALTIME)
    reparam = find_nearest_time_parametrization(msgs, p.joint_trajectory)
    for pp in p.joint_trajectory.points:
        pp.velocities = []
        pp.accelerations = []

    follow_trajectory_wrapper.set_callback_objective(joint_target_release)
    pdb.set_trace()
    #follow_trajectory_wrapper.throwing_motion(p.joint_trajectory)



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

    target.time_from_start.secs = START_MOVE_TIME
    trajectory = JointTrajectory()
    trajectory.points.append(copy.deepcopy(current))
    trajectory.points.append(target)
    trajectory.joint_names = jnames[:6]
    rtr = RobotTrajectory()
    rtr.joint_trajectory = trajectory
    
    fk_wrapper = FKWrapper(jnames[:6])
    pdb.set_trace()

    gripper_wrapper = GripperServiceWrapper()
    follow_trajectory_wrapper = FollowTrajectoryWrapper(gripper_wrapper)

    group.execute(rtr, wait=True)
    #pdb.set_trace()
    #group.stop()
    #gripper_wrapper.gripper_open()
    #pdb.set_trace()
    gripper_wrapper.gripper_close()

    pdb.set_trace()
    execute_trajectory(df)