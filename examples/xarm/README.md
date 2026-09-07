# xArm Real Deployment

This directory contains a real-robot client for xArm deployment that follows the same remote-inference pattern as `examples/libero/main.py`, but uses:

- `xarm-python-sdk` for the 6D arm pose state and arm motion commands
- ROS image topics for the front and wrist cameras
- ROS topics for gripper state and gripper commands
- `scripts/serve_policy.py` as the remote policy server

The client entry point is:

- [main.py](/media/ubuntu/data/home/hy/copy/openpi-main/examples/xarm/main.py)

The policy server entry point is:

- [serve_policy.py](/media/ubuntu/data/home/hy/copy/openpi-main/scripts/serve_policy.py)

## Overview

The deployment is split into two processes:

1. Policy server  
   Loads a trained OpenPI checkpoint and serves websocket inference.
2. xArm client  
   Reads observations, sends them to the server, receives action chunks, and executes them on the real robot.

The observation payload sent to the server is:

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

## Observation Sources

The current implementation uses a mixed source for the observation:

- `observation/image`
  - from ROS topic `/camera_top/color/image_raw`
- `observation/wrist_image`
  - from ROS topic `/camera/color/image_raw`
- `observation/state[:6]`
  - from `XArmAPI.get_position()`
- `observation/state[6]`
  - from ROS topic `/gripper/data`

So the state is:

- `x, y, z, roll, pitch, yaw, gripper`

## Gripper Interfaces

The current gripper interfaces are:

- state topic: `/gripper/data`
- command topic: `/joint_states_single_gripper`
- control topic: `/gripper/ctrl`
- joint name: `center_joint`

This matches the live ROS information you provided:

- `/gripper/data`
- `/gripper/ctrl`
- `/gripper/joint_state`
- `/joint_states_single_gripper`

## Environments

### 1. Server environment

Use your OpenPI runtime environment for the policy server.

Typical launch from the repo root:

```bash
cd /media/ubuntu/data/home/hy/copy/openpi-main
PYTHONPATH=src python scripts/serve_policy.py --env XARM
```

`serve_policy.py` includes an `XARM` mode whose default checkpoint is:

- config: `pi0_xarm_data_heyao_2_incremental_lora`
- checkpoint: `checkpoints/pi0_xarm_data_heyao_2_incremental_lora/xarm_data_heyao_2_incremental_lora_gpu1/50000`

### 2. Client environment

A dedicated conda environment already exists at:

- `/media/ubuntu/data/home/hy/copy/openpi-main/.conda/xarm-client`

Activate it with:

```bash
conda activate /media/ubuntu/data/home/hy/copy/openpi-main/.conda/xarm-client
```

The local `openpi-client` package is used from source through `PYTHONPATH`:

```bash
export PYTHONPATH=/media/ubuntu/data/home/hy/copy/openpi-main/packages/openpi-client/src
```

## ROS Requirements

The client requires a ROS shell where these are available:

- `rospy`
- `sensor_msgs`
- `data_msgs`

So before running the client, source:

1. your ROS installation
2. the workspace that provides the camera topics
3. the workspace that provides `data_msgs`

Example:

```bash
source /path/to/your/ros/setup.bash
source /path/to/your/workspace/devel/setup.bash
```

## Client Launch

From the repo root:

```bash
cd /media/ubuntu/data/home/hy/copy/openpi-main
conda activate /media/ubuntu/data/home/hy/copy/openpi-main/.conda/xarm-client
export PYTHONPATH=/media/ubuntu/data/home/hy/copy/openpi-main/packages/openpi-client/src

python examples/xarm/main.py \
  --host 127.0.0.1 \
  --port 8000 \
  --robot-ip 192.168.10.244 \
  --front-camera-topic /camera_top/color/image_raw \
  --wrist-camera-topic /camera/color/image_raw \
  --gripper-state-topic /gripper/data \
  --gripper-command-topic /joint_states_single_gripper \
  --gripper-ctrl-topic /gripper/ctrl \
  --gripper-joint-name center_joint \
  --prompt "Place the bread onto the plate"
```

If the server is on another machine, replace `--host` with that machine’s IP.

## Important Arguments

The most important runtime arguments in [main.py](/media/ubuntu/data/home/hy/copy/openpi-main/examples/xarm/main.py) are:

- `--robot-ip`
  xArm controller IP.
- `--host`
  Policy server IP or hostname.
- `--port`
  Policy server port.
- `--prompt`
  Task instruction sent to the model.
- `--front-camera-topic`
  Front ROS image topic.
- `--wrist-camera-topic`
  Wrist ROS image topic.
- `--gripper-state-topic`
  ROS topic publishing `data_msgs/Gripper`.
- `--gripper-command-topic`
  ROS topic receiving `sensor_msgs/JointState`.
- `--gripper-ctrl-topic`
  ROS topic receiving `data_msgs/Gripper` control messages.
- `--gripper-joint-name`
  Joint name used inside the command `JointState`. The current value is `center_joint`.
- `--resize-size`
  Input image size sent to the policy server.
- `--replan-steps`
  Number of actions executed from each predicted chunk before replanning.
- `--control-hz`
  Control loop frequency.
- `--position-speed`
  xArm Cartesian speed.
- `--position-acc`
  xArm Cartesian acceleration.
- `--front-rotate-k`
  90-degree rotation count for the front image.
- `--wrist-rotate-k`
  90-degree rotation count for the wrist image.
- `--init-gripper-open`
  Whether to send an initial open command at startup.
- `--gripper-debug`
  Whether to log gripper observation and command mappings.

## Current Safe Defaults

The current client defaults were made more conservative for real hardware:

- `resize_size=224`
- `replan_steps=1`
- `max_steps=200`
- `control_hz=3.0`
- `position_speed=50.0`
- `position_acc=300.0`

These defaults are intended to reduce:

- control jitter
- open-loop drift
- motion abruptness
- gripper debugging ambiguity

## What To Check First

If the robot is not smooth or the gripper still looks wrong, check these first:

1. Verify both camera topics are alive.
2. Verify `/gripper/data` is updating.
3. Verify `/joint_states_single_gripper` is the actual command topic.
4. Verify `center_joint` is still the correct joint name.
5. Watch the client logs for:
   - `gripper observation: ...`
   - `gripper command: ...`

## Example: Remote Inference On Two Machines

### Machine A: policy server

```bash
cd /media/ubuntu/data/home/hy/copy/openpi-main
PYTHONPATH=src python scripts/serve_policy.py --env XARM --port 8000
```

### Machine B: xArm client

```bash
source /path/to/your/ros/setup.bash
source /path/to/your/workspace/devel/setup.bash

cd /media/ubuntu/data/home/hy/copy/openpi-main
conda activate /media/ubuntu/data/home/hy/copy/openpi-main/.conda/xarm-client
export PYTHONPATH=/media/ubuntu/data/home/hy/copy/openpi-main/packages/openpi-client/src

python examples/xarm/main.py \
  --host <SERVER_IP> \
  --port 8000 \
  --robot-ip 192.168.10.244 \
  --front-camera-topic /camera_top/color/image_raw \
  --wrist-camera-topic /camera/color/image_raw \
  --gripper-state-topic /gripper/data \
  --gripper-command-topic /joint_states_single_gripper \
  --gripper-ctrl-topic /gripper/ctrl \
  --gripper-joint-name center_joint \
  --prompt "Place the bread onto the plate"
```

## Safety

This is still a minimal deployment path. Before real execution:

- verify the robot workspace is safe
- verify the reset pose is safe
- verify the gripper range mapping is correct
- start with low speed and low control frequency
- keep `replan_steps=1` until behavior is stable
- test with the arm clear of objects first

The script does not include collision checking or a full safety supervisor.
