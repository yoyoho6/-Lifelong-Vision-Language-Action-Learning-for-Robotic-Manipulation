import collections
import dataclasses
import logging
import pathlib
import threading
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from xarm.wrapper import XArmAPI


DEPLOY_VERSION = "xarm-main-debug-20260518-gripper-print"


def _rotate_image(image: np.ndarray, rotate_k: int) -> np.ndarray:
    rotate_k = rotate_k % 4
    if rotate_k == 0:
        return image
    return np.rot90(image, k=rotate_k)


def _stamp_to_seconds(stamp) -> float:
    return stamp.secs + stamp.nsecs * 1e-9


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta))


def _image_msg_to_rgb(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in {"mono8", "8uc1"}:
        channels = 1
    elif encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    else:
        raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")

    row_size = msg.width * channels
    if msg.step < row_size:
        raise ValueError(f"Invalid ROS image step for encoding {msg.encoding}: {msg.step}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    image_size = msg.height * msg.step
    if data.size < image_size:
        raise ValueError(f"ROS image data is smaller than expected for topic frame: {data.size} < {image_size}")

    rows = data[:image_size].reshape(msg.height, msg.step)
    image = rows[:, :row_size].reshape(msg.height, msg.width, channels)

    if encoding in {"mono8", "8uc1"}:
        return np.ascontiguousarray(np.repeat(image, 3, axis=2))
    if encoding == "rgb8":
        return np.ascontiguousarray(image)
    if encoding == "bgr8":
        return np.ascontiguousarray(image[..., ::-1])
    if encoding == "rgba8":
        return np.ascontiguousarray(image[..., :3])
    return np.ascontiguousarray(image[..., [2, 1, 0]])


class RosImageReader:
    def __init__(self, topic: str, camera_name: str, timeout: float):
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS image reading requires rospy and sensor_msgs. Source your ROS environment before running this client."
            ) from exc

        self._rospy = rospy
        self._topic = topic
        self._camera_name = camera_name
        self._timeout = timeout
        self._lock = threading.Lock()
        self._image: np.ndarray | None = None
        self._frame_index = 0
        self._last_read_frame_index = 0
        self._timestamp = 0.0

        self._subscriber = rospy.Subscriber(
            topic,
            Image,
            self._image_cb,
            queue_size=1,
            buff_size=2**24,
        )

    def close(self) -> None:
        self._subscriber.unregister()

    def read_rgb(self, rotate_k: int) -> np.ndarray:
        deadline = time.time() + self._timeout
        while not self._rospy.is_shutdown():
            with self._lock:
                if self._image is not None and self._frame_index > self._last_read_frame_index:
                    self._last_read_frame_index = self._frame_index
                    return _rotate_image(self._image.copy(), rotate_k)

            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for a new {self._camera_name} image on ROS topic {self._topic}"
                )
            time.sleep(0.001)

        raise RuntimeError(f"ROS shutdown while waiting for {self._camera_name} image on topic {self._topic}")

    def _image_cb(self, msg) -> None:
        try:
            image = _image_msg_to_rgb(msg)
        except ValueError:
            logging.exception("Failed to convert image from ROS topic %s", self._topic)
            return

        with self._lock:
            self._image = image
            self._frame_index += 1
            self._timestamp = _stamp_to_seconds(msg.header.stamp)


class RosGripper:
    def __init__(
        self,
        state_topic: str,
        command_topic: str,
        ctrl_topic: str,
        joint_name: str,
        timeout: float,
        command_frame_id: str,
    ):
        try:
            import rospy
            from data_msgs.msg import Gripper
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError(
                "ROS gripper control requires rospy, sensor_msgs, and data_msgs. "
                "Source ROS and the workspace that provides data_msgs before running this client."
            ) from exc

        self._rospy = rospy
        self._gripper_msg_type = Gripper
        self._joint_state_msg_type = JointState
        self._state_topic = state_topic
        self._command_topic = command_topic
        self._ctrl_topic = ctrl_topic
        self._joint_name = joint_name
        self._timeout = timeout
        self._command_frame_id = command_frame_id
        self._lock = threading.Lock()
        self._distance: float | None = None

        self._state_subscriber = rospy.Subscriber(state_topic, Gripper, self._state_cb, queue_size=1)
        self._command_publisher = rospy.Publisher(command_topic, JointState, queue_size=1)
        self._ctrl_publisher = rospy.Publisher(ctrl_topic, Gripper, queue_size=1)
        fields = getattr(Gripper, "__slots__", [])
        print(f"[gripper] data_msgs/Gripper fields={fields}", flush=True)

    def close(self) -> None:
        self._state_subscriber.unregister()

    def wait_for_publishers(self, timeout: float) -> None:
        print(
            f"[gripper] waiting for subscribers: command_topic={self._command_topic}, ctrl_topic={self._ctrl_topic}, timeout={timeout}",
            flush=True,
        )
        deadline = time.time() + timeout
        while not self._rospy.is_shutdown() and time.time() < deadline:
            if self._command_publisher.get_num_connections() > 0 and self._ctrl_publisher.get_num_connections() > 0:
                print(
                    f"[gripper] subscribers ready: command={self._command_publisher.get_num_connections()}, ctrl={self._ctrl_publisher.get_num_connections()}",
                    flush=True,
                )
                return
            time.sleep(0.02)
        print(
            f"[gripper] subscriber wait timeout: command={self._command_publisher.get_num_connections()}, ctrl={self._ctrl_publisher.get_num_connections()}",
            flush=True,
        )
        logging.warning(
            "gripper publishers may have no subscribers: command_connections=%d ctrl_connections=%d",
            self._command_publisher.get_num_connections(),
            self._ctrl_publisher.get_num_connections(),
        )

    def enable(self) -> None:
        print(f"[gripper] publish enable: topic={self._ctrl_topic}", flush=True)
        msg = self._gripper_msg_type()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = self._command_frame_id
        msg.enable = True
        msg.set_zero = False
        self._ctrl_publisher.publish(msg)

    def read_distance(self) -> float:
        deadline = time.time() + self._timeout
        while not self._rospy.is_shutdown():
            with self._lock:
                if self._distance is not None:
                    return self._distance

            if time.time() >= deadline:
                raise RuntimeError(f"Timed out waiting for gripper distance on ROS topic {self._state_topic}")
            time.sleep(0.001)

        raise RuntimeError(f"ROS shutdown while waiting for gripper distance on topic {self._state_topic}")

    def publish_position(self, position: float, *, repeat: int = 1, interval: float = 0.0, publish_ctrl: bool = True) -> None:
        if not self._joint_name:
            logging.warning("gripper_joint_name is empty; JointState commands may be ignored by the gripper controller")
        repeat = max(int(repeat), 1)
        logging.info(
            "publishing gripper position: topic=%s joint=%s position=%.4f repeat=%d subscribers=%d",
            self._command_topic,
            self._joint_name,
            float(position),
            repeat,
            self._command_publisher.get_num_connections(),
        )
        print(
            f"[gripper] publish position: topic={self._command_topic}, joint={self._joint_name}, position={float(position):.4f}, repeat={repeat}, subscribers={self._command_publisher.get_num_connections()}",
            flush=True,
        )
        for idx in range(repeat):
            msg = self._joint_state_msg_type()
            msg.header.stamp = self._rospy.Time.now()
            msg.name = [self._joint_name]
            msg.position = [float(position)]
            msg.velocity = [0.0]
            msg.effort = [0.0]
            self._command_publisher.publish(msg)
            if publish_ctrl:
                self.publish_ctrl_position(position)
            if idx + 1 < repeat and interval > 0:
                time.sleep(interval)

    def publish_ctrl_position(self, position: float) -> None:
        msg = self._gripper_msg_type()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = self._command_frame_id
        msg.enable = True
        msg.set_zero = False
        for field in ("distance", "position", "pos", "target_position", "target_pos", "command"):
            if hasattr(msg, field):
                setattr(msg, field, float(position))
        logging.info(
            "publishing gripper ctrl position: topic=%s position=%.4f subscribers=%d",
            self._ctrl_topic,
            float(position),
            self._ctrl_publisher.get_num_connections(),
        )
        print(
            f"[gripper] publish ctrl position: topic={self._ctrl_topic}, position={float(position):.4f}, subscribers={self._ctrl_publisher.get_num_connections()}",
            flush=True,
        )
        self._ctrl_publisher.publish(msg)

    def _state_cb(self, msg) -> None:
        if msg.header.frame_id == self._command_frame_id:
            return

        with self._lock:
            self._distance = float(msg.distance)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000

    robot_ip: str = "192.168.10.244"
    prompt: str = "Place the bread onto the plate"

    front_camera_topic: str = "/camera_top/color/image_raw"
    wrist_camera_topic: str = "/camera/color/image_raw"
    ros_node_name: str = "openpi_xarm_client"
    ros_image_timeout: float = 5.0

    resize_size: int = 224
    replan_steps: int = 1
    max_steps: int = 200
    control_hz: float = 3.0

    front_rotate_k: int = 0
    wrist_rotate_k: int = 0

    is_radian: bool = True
    position_speed: float = 50.0
    position_acc: float = 300.0
    tcp_offset: list[float] | None = dataclasses.field(
        default_factory=lambda: [-125.0, -7.77, 265.64, 3.14159, 0.0, 0.0]
    )
    use_tcp_offset: bool = False
    use_relative_pose_execution: bool = True
    max_position_step_mm: float = 3.0
    max_rotation_step_rad: float = 0.05
    lock_orientation: bool = True
    orientation_reference: str = "reset"
    use_relative_set_position_api: bool = False
    use_relative_position_api: bool = False
    use_joint_space_execution: bool = False
    joint_speed: float = 0.3
    joint_acc: float = 1.0
    max_joint_step_rad: float = 0.03
    max_wrist_joint_step_rad: float = 0.015

    gripper_state_topic: str = "/gripper/data"
    gripper_command_topic: str = "/joint_states_single_gripper"
    gripper_ctrl_topic: str = "/gripper/ctrl"
    gripper_joint_name: str = "center_joint"
    gripper_command_frame_id: str = "openpi_xarm_client"
    gripper_state_timeout: float = 5.0
    gripper_close_position: float = 0.0
    gripper_open_position: float = 0.09
    invert_gripper_command: bool = False
    gripper_command_repeat: int = 3
    gripper_command_interval: float = 0.02
    gripper_startup_wait: float = 2.0
    force_gripper_open: bool = False
    init_gripper_open: bool = True
    gripper_debug: bool = True
    stop_on_arm_error: bool = True

    reset_pose: list[float] | None = dataclasses.field(
        default_factory=lambda: [344.4, 1.2, 266.6, -0.02618, 0.0, 0.00873]
    )
    video_out_path: str = "data/xarm/videos"


class XArmRealDeployment:
    def __init__(self, args: Args):
        self._args = args
        self._arm = XArmAPI(args.robot_ip, is_radian=args.is_radian)
        self._init_ros_node(args.ros_node_name)
        self._front_reader = RosImageReader(args.front_camera_topic, "front", args.ros_image_timeout)
        self._wrist_reader = RosImageReader(args.wrist_camera_topic, "wrist", args.ros_image_timeout)
        self._gripper = RosGripper(
            state_topic=args.gripper_state_topic,
            command_topic=args.gripper_command_topic,
            ctrl_topic=args.gripper_ctrl_topic,
            joint_name=args.gripper_joint_name,
            timeout=args.gripper_state_timeout,
            command_frame_id=args.gripper_command_frame_id,
        )

    def setup(self) -> None:
        print("[setup] begin", flush=True)
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        if self._args.use_tcp_offset and self._args.tcp_offset and len(self._args.tcp_offset) == 6:
            code = self._arm.set_tcp_offset(self._args.tcp_offset, is_radian=self._args.is_radian, wait=True)
            if code != 0:
                logging.warning("set_tcp_offset returned code=%s", code)
            else:
                logging.info("Using tcp_offset=%s", self._args.tcp_offset)
        else:
            zero_tcp_offset = [0.0] * 6
            code = self._arm.set_tcp_offset(zero_tcp_offset, is_radian=self._args.is_radian, wait=True)
            if code != 0:
                logging.warning("failed to reset tcp_offset to zero, code=%s", code)
            else:
                logging.info("TCP offset disabled for this run; reset controller tcp_offset to %s", zero_tcp_offset)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        print(
            f"[setup] arm ready, init_gripper_open={self._args.init_gripper_open}, gripper_joint_name={self._args.gripper_joint_name}",
            flush=True,
        )
        if not self._args.gripper_joint_name:
            logging.warning("gripper_joint_name is empty; fill it with the real gripper joint name used by your ROS controller")
        self._gripper.wait_for_publishers(self._args.gripper_startup_wait)
        self._gripper.enable()
        time.sleep(0.2)
        if self._args.init_gripper_open:
            print(f"[setup] opening gripper to {self._args.gripper_open_position}", flush=True)
            self._gripper.publish_position(
                self._args.gripper_open_position,
                repeat=max(self._args.gripper_command_repeat, 10),
                interval=self._args.gripper_command_interval,
            )
            time.sleep(0.3)

    def disconnect(self) -> None:
        self._front_reader.close()
        self._wrist_reader.close()
        self._gripper.close()
        self._arm.disconnect()

    def reset(self) -> None:
        if self._args.reset_pose and len(self._args.reset_pose) == 6:
            code = self._arm.set_position(
                x=self._args.reset_pose[0],
                y=self._args.reset_pose[1],
                z=self._args.reset_pose[2],
                roll=self._args.reset_pose[3],
                pitch=self._args.reset_pose[4],
                yaw=self._args.reset_pose[5],
                speed=self._args.position_speed,
                mvacc=self._args.position_acc,
                wait=True,
                is_radian=self._args.is_radian,
            )
            if code != 0:
                logging.warning("Reset pose command returned code=%s", code)

    def get_observation(self) -> tuple[dict, np.ndarray]:
        front = self._front_reader.read_rgb(self._args.front_rotate_k)
        wrist = self._wrist_reader.read_rgb(self._args.wrist_rotate_k)

        front = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(front, self._args.resize_size, self._args.resize_size)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, self._args.resize_size, self._args.resize_size)
        )

        state = self._get_state()
        obs = {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": self._args.prompt,
        }
        return obs, front

    def apply_action(self, action: np.ndarray) -> None:
        target_pose = np.asarray(action[:6], dtype=np.float32)
        gripper_position = float(action[6])
        logging.info("action=%s", np.array2string(np.asarray(action), precision=4))

        code, current_pose = self._arm.get_position(is_radian=self._args.is_radian)
        if code != 0:
            raise RuntimeError(f"get_position failed before command with code={code}")
        current_pose = np.asarray(current_pose[:6], dtype=np.float32)

        if self._args.lock_orientation:
            target_pose[3:6] = self._get_locked_orientation(current_pose)
            logging.info(
                "locked orientation: reference=%s rpy=%s",
                self._args.orientation_reference,
                np.array2string(target_pose[3:6], precision=4),
            )

        command_pose = target_pose
        clipped_delta = None
        if self._args.use_relative_pose_execution:
            delta = target_pose - current_pose
            max_steps = np.asarray(
                [
                    self._args.max_position_step_mm,
                    self._args.max_position_step_mm,
                    self._args.max_position_step_mm,
                    self._args.max_rotation_step_rad,
                    self._args.max_rotation_step_rad,
                    self._args.max_rotation_step_rad,
                ],
                dtype=np.float32,
            )
            clipped_delta = np.clip(delta, -max_steps, max_steps)
            command_pose = current_pose + clipped_delta
            logging.info(
                "pose command: current=%s target=%s clipped_delta=%s command=%s",
                np.array2string(current_pose, precision=4),
                np.array2string(target_pose, precision=4),
                np.array2string(clipped_delta, precision=4),
                np.array2string(command_pose, precision=4),
            )

        command_api = "unknown"
        if self._args.use_joint_space_execution:
            command_api = "set_servo_angle"
            code = self._execute_joint_space_pose(command_pose)
        elif self._args.use_relative_set_position_api and clipped_delta is not None and self._args.lock_orientation:
            command_api = "set_tool_position"
            logging.info(
                "tool-frame command: dx=%.4f dy=%.4f dz=%.4f",
                float(clipped_delta[0]),
                float(clipped_delta[1]),
                float(clipped_delta[2]),
            )
            code = self._arm.set_tool_position(
                x=float(clipped_delta[0]),
                y=float(clipped_delta[1]),
                z=float(clipped_delta[2]),
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                speed=self._args.position_speed,
                mvacc=self._args.position_acc,
                wait=False,
                is_radian=self._args.is_radian,
            )
        elif self._args.use_relative_position_api and clipped_delta is not None:
            command_api = "set_position_relative"
            roll_delta = 0.0 if self._args.lock_orientation else float(clipped_delta[3])
            pitch_delta = 0.0 if self._args.lock_orientation else float(clipped_delta[4])
            yaw_delta = 0.0 if self._args.lock_orientation else float(clipped_delta[5])
            logging.info(
                "relative pose command: dx=%.4f dy=%.4f dz=%.4f droll=%.4f dpitch=%.4f dyaw=%.4f",
                float(clipped_delta[0]),
                float(clipped_delta[1]),
                float(clipped_delta[2]),
                roll_delta,
                pitch_delta,
                yaw_delta,
            )
            code = self._arm.set_position(
                x=float(clipped_delta[0]),
                y=float(clipped_delta[1]),
                z=float(clipped_delta[2]),
                roll=roll_delta,
                pitch=pitch_delta,
                yaw=yaw_delta,
                speed=self._args.position_speed,
                mvacc=self._args.position_acc,
                wait=False,
                relative=True,
                is_radian=self._args.is_radian,
            )
        else:
            command_api = "set_position"
            code = self._arm.set_position(
                x=float(command_pose[0]),
                y=float(command_pose[1]),
                z=float(command_pose[2]),
                roll=float(command_pose[3]),
                pitch=float(command_pose[4]),
                yaw=float(command_pose[5]),
                speed=self._args.position_speed,
                mvacc=self._args.position_acc,
                wait=False,
                is_radian=self._args.is_radian,
            )
        if code != 0:
            logging.warning("%s returned code=%s", command_api, code)
            self._log_arm_status(target_pose=target_pose, api_code=code)
            if self._args.stop_on_arm_error:
                raise RuntimeError(f"xArm {command_api} failed with code={code}")

        if self._args.force_gripper_open:
            gripper_position = self._args.gripper_open_position
            logging.info("force_gripper_open=True; overriding gripper command to %.4f", gripper_position)
        gripper_position = self._normalize_gripper_command(gripper_position)
        if self._args.gripper_debug:
            logging.info(
                "gripper command: position=%.4f close=%.4f open=%.4f",
                gripper_position,
                self._args.gripper_close_position,
                self._args.gripper_open_position,
            )
        self._gripper.publish_position(
            gripper_position,
            repeat=self._args.gripper_command_repeat,
            interval=self._args.gripper_command_interval,
        )

    def _get_state(self) -> np.ndarray:
        code, pose = self._arm.get_position(is_radian=self._args.is_radian)
        if code != 0:
            raise RuntimeError(f"get_position failed with code={code}")

        gripper_position = self._gripper.read_distance()
        if self._args.gripper_debug:
            logging.info(
                "gripper observation: position=%.4f",
                gripper_position,
            )
        return np.asarray([*pose[:6], float(gripper_position)], dtype=np.float32)

    def _get_locked_orientation(self, current_pose: np.ndarray) -> np.ndarray:
        reference = self._args.orientation_reference.lower()
        if reference == "current":
            return current_pose[3:6].copy()
        if reference == "reset":
            if not self._args.reset_pose or len(self._args.reset_pose) != 6:
                raise ValueError("orientation_reference='reset' requires reset_pose with 6 values")
            return np.asarray(self._args.reset_pose[3:6], dtype=np.float32)
        if reference == "model":
            raise ValueError("orientation_reference='model' should be used with --no-lock-orientation")
        raise ValueError("orientation_reference must be one of: reset, current, model")

    def _normalize_gripper_command(self, position: float) -> float:
        close_position = float(self._args.gripper_close_position)
        open_position = float(self._args.gripper_open_position)
        low = min(close_position, open_position)
        high = max(close_position, open_position)
        clipped = float(np.clip(position, low, high))
        if self._args.invert_gripper_command:
            clipped = close_position + open_position - clipped
        if clipped != position:
            logging.info("clipped gripper command from %.4f to %.4f", position, clipped)
        return clipped

    def _execute_joint_space_pose(self, pose: np.ndarray) -> int:
        ik_code, ik_angles = self._arm.get_inverse_kinematics(
            pose.tolist(),
            input_is_radian=self._args.is_radian,
            return_is_radian=self._args.is_radian,
        )
        if ik_code != 0:
            logging.warning("get_inverse_kinematics returned code=%s for pose=%s", ik_code, pose.tolist())
            self._log_arm_status(target_pose=pose, api_code=ik_code)
            return ik_code

        current_joint_code, current_joints = self._arm.get_servo_angle(is_radian=self._args.is_radian)
        if current_joint_code != 0:
            logging.warning("get_servo_angle returned code=%s", current_joint_code)
            self._log_arm_status(target_pose=pose, api_code=current_joint_code)
            return current_joint_code

        current_joints = np.asarray(current_joints, dtype=np.float32)
        ik_angles = np.asarray(ik_angles[: len(current_joints)], dtype=np.float32)
        joint_delta = _wrap_angle_delta(ik_angles - current_joints)

        max_joint_steps = np.full(len(current_joints), self._args.max_joint_step_rad, dtype=np.float32)
        if len(current_joints) >= 3:
            max_joint_steps[-3:] = np.minimum(max_joint_steps[-3:], self._args.max_wrist_joint_step_rad)
        clipped_joint_delta = np.clip(joint_delta, -max_joint_steps, max_joint_steps)
        command_joints = current_joints + clipped_joint_delta

        logging.info(
            "joint command: current=%s ik=%s clipped_delta=%s command=%s",
            np.array2string(current_joints, precision=4),
            np.array2string(ik_angles, precision=4),
            np.array2string(clipped_joint_delta, precision=4),
            np.array2string(command_joints, precision=4),
        )
        return self._arm.set_servo_angle(
            angle=command_joints.tolist(),
            speed=self._args.joint_speed,
            mvacc=self._args.joint_acc,
            wait=False,
            is_radian=self._args.is_radian,
        )

    def _init_ros_node(self, node_name: str) -> None:
        try:
            import rospy
        except ImportError as exc:
            raise RuntimeError("ROS image reading requires rospy. Source your ROS environment before running this client.") from exc

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

    def _log_arm_status(self, *, target_pose: np.ndarray | None = None, api_code: int | None = None) -> None:
        try:
            err_warn = self._arm.get_err_warn_code()
        except Exception as exc:
            err_warn = f"<failed: {exc}>"
        try:
            tcp_offset = self._arm.tcp_offset
        except Exception as exc:
            tcp_offset = f"<failed: {exc}>"

        logging.error(
            "xArm status after motion failure: api_code=%s state=%s mode=%s error_code=%s warn_code=%s has_error=%s has_warn=%s err_warn=%s tcp_offset=%s target_pose=%s",
            api_code,
            getattr(self._arm, "state", None),
            getattr(self._arm, "mode", None),
            getattr(self._arm, "error_code", None),
            getattr(self._arm, "warn_code", None),
            getattr(self._arm, "has_error", None),
            getattr(self._arm, "has_warn", None),
            err_warn,
            tcp_offset,
            None if target_pose is None else np.asarray(target_pose).tolist(),
        )


def main(args: Args) -> None:
    print(f"[xarm-main] file={__file__}", flush=True)
    print(f"[xarm-main] version={DEPLOY_VERSION}", flush=True)
    print(
        "[xarm-main] gripper defaults: "
        f"state_topic={args.gripper_state_topic}, command_topic={args.gripper_command_topic}, "
        f"ctrl_topic={args.gripper_ctrl_topic}, joint={args.gripper_joint_name}, "
        f"init_open={args.init_gripper_open}, open={args.gripper_open_position}, "
        f"force_open={args.force_gripper_open}",
        flush=True,
    )
    print(
        "[xarm-main] motion defaults: "
        f"use_tcp_offset={args.use_tcp_offset}, orientation_reference={args.orientation_reference}, "
        f"use_joint_space_execution={args.use_joint_space_execution}, "
        f"use_relative_set_position_api={args.use_relative_set_position_api}, "
        f"use_relative_position_api={args.use_relative_position_api}, "
        f"max_position_step_mm={args.max_position_step_mm}, "
        f"position_speed={args.position_speed}, position_acc={args.position_acc}",
        flush=True,
    )
    policy_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Server metadata: %s", policy_client.get_server_metadata())

    deployment = XArmRealDeployment(args)
    print("[xarm-main] deployment constructed, calling setup()", flush=True)
    deployment.setup()
    print("[xarm-main] setup() finished, calling reset()", flush=True)
    deployment.reset()
    print("[xarm-main] reset() finished, entering rollout loop", flush=True)

    action_plan = collections.deque()
    replay_images: list[np.ndarray] = []
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    try:
        for step in range(args.max_steps):
            step_start = time.time()
            obs, replay_image = deployment.get_observation()
            replay_images.append(replay_image)

            if not action_plan:
                action_chunk = np.asarray(policy_client.infer(obs)["actions"], dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                    raise RuntimeError(f"Unexpected action chunk shape: {action_chunk.shape}")
                action_plan.extend(action_chunk[: args.replan_steps])

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            deployment.apply_action(action)

            elapsed = time.time() - step_start
            target_dt = 1.0 / args.control_hz
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            logging.info("step=%d action=%s", step, np.array2string(action, precision=4))
    finally:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = pathlib.Path(args.video_out_path) / f"xarm_rollout_{timestamp}.mp4"
        if replay_images:
            imageio.mimwrite(video_path, replay_images, fps=max(int(args.control_hz), 1))
            logging.info("Saved rollout video to %s", video_path)
        deployment.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
