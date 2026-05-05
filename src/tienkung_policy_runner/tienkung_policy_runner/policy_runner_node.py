from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from bodyctrl_msgs.msg import CmdMotorCtrl, CmdSetMotorPosition
from bodyctrl_msgs.msg import Imu as BodyImu
from bodyctrl_msgs.msg import MotorStatusMsg
from rclpy.node import Node
from sensor_msgs.msg import Joy

from tienkung_interfaces.msg import ControlMode, MotionReference

from .fsm import ControlMode as LocalControlMode, PolicyFSM
from .manifest_loader import load_manifest, validate_manifest_against_contract
from .obs_builder import TienkungObservationBuilder, postprocess_action
from .onnx_runtime import OnnxPolicyRunner
from .robot_contract import DEFAULT_MIMIC_OBS_TIENKUNG, get_robot_contract
from .robot_io import JointMap, RobotIO, default_gains


class PolicyRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("tienkung_policy_runner")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("manifest_path", "")
        self.declare_parameter("policy_hz", 50.0)
        self.declare_parameter("zero_duration_sec", 2.0)
        self.declare_parameter("state_timeout_sec", 0.1)
        self.declare_parameter("motion_timeout_sec", 0.1)
        self.declare_parameter("require_motion_source", False)
        self.declare_parameter("motion_reference_topic", "/tienkung/motion_reference")
        self.declare_parameter("control_mode_topic", "/tienkung/control_mode")
        self.declare_parameter("leg_status_topic", "/leg/status")
        self.declare_parameter("arm_status_topic", "/arm/status")
        self.declare_parameter("imu_status_topic", "/imu/status")
        self.declare_parameter("joy_topic", "/sbus_data")
        self.declare_parameter("leg_command_topic", "/leg/cmd_ctrl")
        self.declare_parameter("arm_command_topic", "/arm/cmd_ctrl")
        self.declare_parameter("waist_command_topic", "/waist/cmd_pos")

        self.contract = get_robot_contract("tienkung")

        manifest_path = str(self.get_parameter("manifest_path").value)
        if manifest_path:
            manifest = load_manifest(manifest_path)
            validate_manifest_against_contract(manifest, self.contract)

        policy_path = str(self.get_parameter("policy_path").value)
        if not policy_path:
            raise ValueError("policy_path is required")
        self.policy = OnnxPolicyRunner(policy_path, str(self.get_parameter("device").value))

        joint_map_path = Path(__file__).resolve().parents[1] / "config" / "joint_map.yaml"
        if not joint_map_path.exists():
            joint_map_path = Path(__file__).resolve().parents[2] / "config" / "joint_map.yaml"

        self.joint_map = JointMap.from_yaml(str(joint_map_path))
        self.robot_io = RobotIO(self.joint_map, self.contract)
        self.obs_builder = TienkungObservationBuilder(self.contract)
        self.fsm = PolicyFSM(float(self.get_parameter("zero_duration_sec").value))
        self.kp, self.kd = default_gains(self.contract)
        self.zero_traj_duration_sec = max(
            1e-3, float(self.get_parameter("zero_duration_sec").value)
        )

        self.last_action = np.zeros(self.contract.num_actions, dtype=np.float32)
        # action_delay_steps: match training action_delay_range=[0,3]; 1 step = 20ms
        self._action_delay_steps = 1
        self._action_buf: deque[np.ndarray] = deque(
            [np.zeros(self.contract.num_actions, dtype=np.float32)] * (self._action_delay_steps + 1),
            maxlen=self._action_delay_steps + 1,
        )
        self.latest_motion_reference = DEFAULT_MIMIC_OBS_TIENKUNG.copy()
        self.last_motion_reference_time_sec = 0.0
        self.zero_start_time_sec: float | None = None
        self.zero_start_dof_pos = self.contract.default_dof_pos.copy()
        self.prev_mode = LocalControlMode.STOP
        self._last_state_ready = False
        self._last_motion_ready = False
        self._last_wait_log_time_sec = 0.0
        self._leg_status_seen = False
        self._arm_status_seen = False
        self._imu_seen = False
        self._joy_seen = False
        self.state_timeout_sec = float(self.get_parameter("state_timeout_sec").value)
        self.motion_timeout_sec = float(self.get_parameter("motion_timeout_sec").value)
        self.require_motion_source = bool(self.get_parameter("require_motion_source").value)
        self._motion_fallback_logged = False

        self.get_logger().info(
            "Policy runner starting with "
            f"policy_path='{policy_path}', manifest_path='{manifest_path}', "
            f"device='{self.get_parameter('device').value}', policy_hz={float(self.get_parameter('policy_hz').value):.1f}, "
            f"state_timeout_sec={self.state_timeout_sec:.3f}, motion_timeout_sec={self.motion_timeout_sec:.3f}"
        )
        self.get_logger().info(
            "Subscribed topics: "
            f"motion_ref={self.get_parameter('motion_reference_topic').value}, "
            f"leg_status={self.get_parameter('leg_status_topic').value}, "
            f"arm_status={self.get_parameter('arm_status_topic').value}, "
            f"imu={self.get_parameter('imu_status_topic').value}, "
            f"joy={self.get_parameter('joy_topic').value}"
        )
        self.get_logger().info(
            "Publishing topics: "
            f"control_mode={self.get_parameter('control_mode_topic').value}, "
            f"leg_cmd={self.get_parameter('leg_command_topic').value}, "
            f"arm_cmd={self.get_parameter('arm_command_topic').value}, "
            f"waist_cmd={self.get_parameter('waist_command_topic').value}"
        )

        self.create_subscription(
            MotionReference,
            str(self.get_parameter("motion_reference_topic").value),
            self._motion_reference_callback,
            10,
        )
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("leg_status_topic").value),
            self._leg_status_callback,
            100,
        )
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("arm_status_topic").value),
            self._arm_status_callback,
            100,
        )
        self.create_subscription(
            BodyImu,
            str(self.get_parameter("imu_status_topic").value),
            self._imu_callback,
            100,
        )
        self.create_subscription(
            Joy,
            str(self.get_parameter("joy_topic").value),
            self._joy_callback,
            100,
        )

        self.control_mode_pub = self.create_publisher(
            ControlMode,
            str(self.get_parameter("control_mode_topic").value),
            10,
        )
        self.leg_cmd_pub = self.create_publisher(
            CmdMotorCtrl,
            str(self.get_parameter("leg_command_topic").value),
            10,
        )
        self.arm_cmd_pub = self.create_publisher(
            CmdMotorCtrl,
            str(self.get_parameter("arm_command_topic").value),
            10,
        )
        self.waist_cmd_pub = self.create_publisher(
            CmdSetMotorPosition,
            str(self.get_parameter("waist_command_topic").value),
            10,
        )
        self.timer = self.create_timer(1.0 / float(self.get_parameter("policy_hz").value), self._tick)

    def _stamp_to_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _motion_reference_callback(self, msg: MotionReference) -> None:
        body = np.asarray(msg.body_mimic, dtype=np.float32)
        if body.shape[0] == self.contract.n_mimic_obs:
            if self.last_motion_reference_time_sec <= 0.0:
                self.get_logger().info(
                    f"First motion reference received on {self.get_parameter('motion_reference_topic').value} "
                    f"(dim={body.shape[0]})"
                )
            self.latest_motion_reference = body
            self.last_motion_reference_time_sec = self._stamp_to_sec()
        else:
            self.get_logger().warning(
                f"Ignoring motion reference with unexpected dim={body.shape[0]}, expected {self.contract.n_mimic_obs}"
            )

    def _leg_status_callback(self, msg: MotorStatusMsg) -> None:
        self.robot_io.ingest_leg_status(msg, self._stamp_to_sec())
        if not self._leg_status_seen:
            self._leg_status_seen = True
            self.get_logger().info(
                f"First leg status received on {self.get_parameter('leg_status_topic').value} "
                f"(motors={len(getattr(msg, 'status', []))})"
            )

    def _arm_status_callback(self, msg: MotorStatusMsg) -> None:
        self.robot_io.ingest_arm_status(msg, self._stamp_to_sec())
        if not self._arm_status_seen:
            self._arm_status_seen = True
            self.get_logger().info(
                f"First arm status received on {self.get_parameter('arm_status_topic').value} "
                f"(motors={len(getattr(msg, 'status', []))})"
            )

    def _imu_callback(self, msg: BodyImu) -> None:
        self.robot_io.ingest_imu(msg, self._stamp_to_sec())
        if not self._imu_seen:
            self._imu_seen = True
            self.get_logger().info(
                f"First IMU status received on {self.get_parameter('imu_status_topic').value}"
            )

    def _joy_callback(self, msg: Joy) -> None:
        self.robot_io.ingest_joy(msg)
        if not self._joy_seen:
            self._joy_seen = True
            self.get_logger().info(
                f"First joystick message received on {self.get_parameter('joy_topic').value} "
                f"(axes={len(getattr(msg, 'axes', []))}, buttons={len(getattr(msg, 'buttons', []))})"
            )

    @staticmethod
    def _mode_name(mode: LocalControlMode) -> str:
        return mode.name

    def _publish_control_mode(self, mode: LocalControlMode) -> None:
        msg = ControlMode()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = int(mode.value)
        self.control_mode_pub.publish(msg)

    @staticmethod
    def _quintic_blend(alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)

    def _tick(self) -> None:
        now_sec = self._stamp_to_sec()
        state = self.robot_io.snapshot()
        joy = self.robot_io.last_joy
        if joy is None:
            return

        state_ready = (now_sec - state.timestamp_sec) <= self.state_timeout_sec and state.timestamp_sec > 0.0
        if self.require_motion_source:
            motion_ready = (now_sec - self.last_motion_reference_time_sec) <= self.motion_timeout_sec and self.last_motion_reference_time_sec > 0.0
        else:
            motion_ready = True
        if state_ready != self._last_state_ready:
            if state_ready:
                self.get_logger().info("State inputs are ready")
            else:
                age = now_sec - state.timestamp_sec if state.timestamp_sec > 0.0 else float("inf")
                age_str = f"{age:.3f}s" if np.isfinite(age) else "never"
                self.get_logger().warning(f"State inputs timed out or missing (age={age_str})")
            self._last_state_ready = state_ready

        if motion_ready != self._last_motion_ready:
            if motion_ready:
                self.get_logger().info("Motion reference is ready")
            else:
                age = (
                    now_sec - self.last_motion_reference_time_sec
                    if self.last_motion_reference_time_sec > 0.0
                    else float("inf")
                )
                age_str = f"{age:.3f}s" if np.isfinite(age) else "never"
                self.get_logger().warning(f"Motion reference timed out or missing (age={age_str})")
            self._last_motion_ready = motion_ready

        mode = self.fsm.update(now_sec, joy, state_ready, motion_ready, True)
        if joy.requested_mode == LocalControlMode.POLICY and mode != LocalControlMode.POLICY:
            if (now_sec - self._last_wait_log_time_sec) >= 1.0:
                reasons = []
                if not state_ready:
                    reasons.append("state not ready")
                if not motion_ready:
                    reasons.append("motion reference not ready")
                wait_reason = ", ".join(reasons) if reasons else "zero transition not finished"
                self.get_logger().warning(
                    f"POLICY requested but not entered yet: {wait_reason}"
                )
                self._last_wait_log_time_sec = now_sec

        self._publish_control_mode(mode)
        if mode != self.prev_mode:
            self.get_logger().info(
                f"Control mode changed: {self._mode_name(self.prev_mode)} -> {self._mode_name(mode)}"
            )
            if mode == LocalControlMode.POLICY and not self.require_motion_source and self.last_motion_reference_time_sec <= 0.0 and not self._motion_fallback_logged:
                self.get_logger().info(
                    "No motion reference received — using DEFAULT_MIMIC_OBS_TIENKUNG as motion fallback"
                )
                self._motion_fallback_logged = True

        if mode == LocalControlMode.ZERO and self.prev_mode != LocalControlMode.ZERO:
            self.zero_start_time_sec = now_sec
            self.zero_start_dof_pos = state.dof_pos.copy()
        elif mode != LocalControlMode.ZERO:
            self.zero_start_time_sec = None

        if mode == LocalControlMode.STOP:
            target = state.dof_pos.copy()
        elif mode == LocalControlMode.ZERO:
            if self.zero_start_time_sec is None:
                self.zero_start_time_sec = now_sec
                self.zero_start_dof_pos = state.dof_pos.copy()

            alpha = np.clip(
                (now_sec - self.zero_start_time_sec) / self.zero_traj_duration_sec,
                0.0,
                1.0,
            )
            blend = self._quintic_blend(float(alpha))
            target = (
                self.zero_start_dof_pos
                + (self.contract.default_dof_pos - self.zero_start_dof_pos) * blend
            ).astype(np.float32)
            self.last_action = np.zeros(self.contract.num_actions, dtype=np.float32)
        else:
            obs = self.obs_builder.build_observation(
                self.latest_motion_reference,
                state.ang_vel,
                state.rpy,
                state.dof_pos,
                state.dof_vel,
                self.last_action,
            )
            raw_action = np.asarray(self.policy.infer(obs)[0], dtype=np.float32)
            self._action_buf.append(raw_action)
            delayed_action = self._action_buf[0]
            self.last_action = delayed_action
            target = postprocess_action(delayed_action, self.contract)

        if mode == LocalControlMode.STOP:
            self.last_action = np.zeros(self.contract.num_actions, dtype=np.float32)
            self._action_buf.extend(
                [np.zeros(self.contract.num_actions, dtype=np.float32)] * (self._action_delay_steps + 1)
            )

        leg_msg, arm_msg, waist_msg = self.robot_io.build_ros_messages(
            self,
            self.robot_io.build_command_dict(target, self.kp, self.kd),
        )
        self.leg_cmd_pub.publish(leg_msg)
        self.arm_cmd_pub.publish(arm_msg)
        self.waist_cmd_pub.publish(waist_msg)
        self.prev_mode = mode


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PolicyRunnerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
