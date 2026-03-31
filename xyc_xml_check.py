import mujoco
import mujoco.viewer
import numpy as np
import os
import re
import tempfile
import heapq
import argparse
from collections import deque
from ipdb import set_trace as xxxx

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m", choices=["eepos", "joint"], default="eepos",
                    help="Control mode: eepos or joint")
args = parser.parse_args()

# load mujoco model
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "assets/demo_fourCable_withSteel_withSensor_cylinder.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# confirm control mode
if args.mode == "eepos":
    print("Control mode: EE POS")
elif args.mode == "joint":
    print("Control mode: JOINT")
else:
    raise ValueError(f"Invalid control mode: {args.mode}")

# eepos control mode
if args.mode == "eepos":
    # start mujoco viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    while True:
        # input target end-effector position
        print("Enter the target end-effector position (x y z):")
        target_pos = input().split()
        target_pos = [float(i) for i in target_pos]
        print(f"Target end-effector position: {target_pos}")
        # control the robot to the target end-effector position
        # data.mocap_pos[0][0] = target_pos[0]
        # data.mocap_pos[0][1] = target_pos[1]
        # data.mocap_pos[0][2] = target_pos[2]
        # xxxx()
        # data.mocap_pos[0] = target_pos
        data.xpos[model.body('link7').id][0] = target_pos[0]
        data.xpos[model.body('link7').id][1] = target_pos[1]
        data.xpos[model.body('link7').id][2] = target_pos[2]
        data.xquat[model.body('link7').id] = np.array([0, 0, 1, 0])
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        # print robot state
        print(f"Robot state: {data.qpos[:7]}")
        print(f"MocapPos : {data.mocap_pos}")
        print(f"EEPos : {data.xpos[model.body('link7').id]}")
        print(f"EEQuat : {data.xquat[model.body('link7').id]}")
        print("--------------------------------")

# joint control mode
if args.mode == "joint":
    # start mujoco viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    while True:
        # input target joint angles
        print("Enter the target joint angles (j1 j2 j3 j4 j5 j6 j7):")
        target_joint_angles = input().split()
        target_joint_angles = [float(i) for i in target_joint_angles]
        print(f"Target joint angles: {target_joint_angles}")
        # control the robot to the target joint angles
        data.qpos[:7] = np.array(target_joint_angles)
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        # print robot state
        print(f"Robot state: {data.qpos[:7]}")
        print(f"MocapPos : {data.mocap_pos}")
        print(f"EEPos : {data.xpos[model.body('link7').id]}")