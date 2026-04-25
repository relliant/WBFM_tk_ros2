"""input_monitor_node.py

A diagnostic ROS 2 node that subscribes to **all topics consumed by the
policy runner** and prints a live summary at a configurable rate.

Topics monitored
----------------
Input (model observations):
  /leg/status          bodyctrl_msgs/MotorStatusMsg   -- leg joint pos/vel/torque
  /arm/status          bodyctrl_msgs/MotorStatusMsg   -- arm joint pos/vel/torque
  /imu/status          bodyctrl_msgs/Imu              -- RPY + angular velocity
  /tienkung/motion_reference  tienkung_interfaces/MotionReference  -- body_mimic 26-dim
  /sbus_data           sensor_msgs/Joy                -- joystick axes/buttons

Output (for reference):
  /tienkung/control_mode  tienkung_interfaces/ControlMode

Usage
-----
  ros2 run tienkung_policy_runner input_monitor_node
  ros2 run tienkung_policy_runner input_monitor_node \
      --ros-args -p print_hz:=2.0 -p verbose:=true
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np
import rclpy
from bodyctrl_msgs.msg import Imu as BodyImu
from bodyctrl_msgs.msg import MotorStatusMsg
from rclpy.node import Node
from sensor_msgs.msg import Joy

from tienkung_interfaces.msg import ControlMode, MotionReference

from .robot_contract import DEFAULT_MIMIC_OBS_TIENKUNG, get_robot_contract
from .robot_io import JointMap


# ── formatting constants ────────────────────────────────────────────────────
_SEP   = "=" * 72
_SEP2  = "-" * 72

_JOINT_NAMES = [
    "hip_roll_l",   "hip_pitch_l",  "hip_yaw_l",    "knee_pitch_l",
    "ankle_pitch_l","ankle_roll_l",
    "hip_roll_r",   "hip_pitch_r",  "hip_yaw_r",    "knee_pitch_r",
    "ankle_pitch_r","ankle_roll_r",
    "shld_pitch_l", "shld_roll_l",  "shld_yaw_l",   "elbow_l",
    "shld_pitch_r", "shld_roll_r",  "shld_yaw_r",   "elbow_r",
]


def _fmt_vec(arr: np.ndarray, fmt: str = "+.4f") -> str:
    return "[" + "  ".join(f"{v:{fmt}}" for v in arr) + "]"


def _fmt_joint_table(values: np.ndarray, label: str, cols: int = 6) -> list[str]:
    """Print joint values in rows of `cols` with aligned column-header names above."""
    lines = [f"  {label}:"]
    n = len(values)
    for row_start in range(0, n, cols):
        chunk = values[row_start:row_start + cols]
        names = _JOINT_NAMES[row_start:row_start + cols]
        name_str = "".join(f"{nm:>16s}" for nm in names)
        val_str  = "".join(f"{v:>+16.4f}" for v in chunk)
        lines.append(f"    {name_str}")
        lines.append(f"    {val_str}")
    return lines


class InputMonitorNode(Node):
    """订阅策略运行器所有输入 topic，并定期打印摘要信息。"""

    def __init__(self) -> None:
        super().__init__("tienkung_input_monitor")

        # ---------- parameters ----------
        self.declare_parameter("print_hz", 1.0)
        self.declare_parameter("verbose", False)
        self.declare_parameter("leg_status_topic", "/leg/status")
        self.declare_parameter("arm_status_topic", "/arm/status")
        self.declare_parameter("imu_status_topic", "/imu/status")
        self.declare_parameter("motion_reference_topic", "/tienkung/motion_reference")
        self.declare_parameter("joy_topic", "/sbus_data")
        self.declare_parameter("control_mode_topic", "/tienkung/control_mode")

        self._verbose: bool = bool(self.get_parameter("verbose").value)
        self._contract = get_robot_contract("tienkung")

        # JointMap: CAN ID → policy index (same file as policy runner)
        from pathlib import Path
        _jmap_path = Path(__file__).resolve().parents[1] / "config" / "joint_map.yaml"
        if not _jmap_path.exists():
            _jmap_path = Path(__file__).resolve().parents[2] / "config" / "joint_map.yaml"
        self._joint_map = JointMap.from_yaml(str(_jmap_path))

        # ---------- internal state ----------
        self._lock = threading.Lock()

        # Leg status
        self._leg_recv_count: int = 0
        self._leg_last_stamp: float = 0.0
        self._leg_pos: np.ndarray = np.zeros(self._contract.num_actions, dtype=np.float32)
        self._leg_vel: np.ndarray = np.zeros(self._contract.num_actions, dtype=np.float32)
        self._leg_torque: np.ndarray = np.zeros(self._contract.num_actions, dtype=np.float32)

        # Arm status
        self._arm_recv_count: int = 0
        self._arm_last_stamp: float = 0.0

        # IMU
        self._imu_recv_count: int = 0
        self._imu_last_stamp: float = 0.0
        self._rpy: np.ndarray = np.zeros(3, dtype=np.float32)
        self._ang_vel: np.ndarray = np.zeros(3, dtype=np.float32)

        # Motion reference
        self._motion_recv_count: int = 0
        self._motion_last_stamp: float = 0.0
        self._motion_ref: np.ndarray = DEFAULT_MIMIC_OBS_TIENKUNG.copy()

        # Joy
        self._joy_recv_count: int = 0
        self._joy_last_stamp: float = 0.0
        self._joy_axes: list[float] = []
        self._joy_buttons: list[int] = []

        # Control mode (output)
        self._mode_recv_count: int = 0
        self._latest_mode: Optional[int] = None

        # ---------- subscriptions ----------
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("leg_status_topic").value),
            self._on_leg_status,
            100,
        )
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("arm_status_topic").value),
            self._on_arm_status,
            100,
        )
        self.create_subscription(
            BodyImu,
            str(self.get_parameter("imu_status_topic").value),
            self._on_imu,
            100,
        )
        self.create_subscription(
            MotionReference,
            str(self.get_parameter("motion_reference_topic").value),
            self._on_motion_reference,
            10,
        )
        self.create_subscription(
            Joy,
            str(self.get_parameter("joy_topic").value),
            self._on_joy,
            100,
        )
        self.create_subscription(
            ControlMode,
            str(self.get_parameter("control_mode_topic").value),
            self._on_control_mode,
            10,
        )

        # ---------- print timer ----------
        period = 1.0 / max(float(self.get_parameter("print_hz").value), 0.1)
        self.create_timer(period, self._print_summary)

        self.get_logger().info(
            "[InputMonitor] Monitoring started. Topics:\n"
            f"  leg  : {self.get_parameter('leg_status_topic').value}\n"
            f"  arm  : {self.get_parameter('arm_status_topic').value}\n"
            f"  imu  : {self.get_parameter('imu_status_topic').value}\n"
            f"  motion_ref : {self.get_parameter('motion_reference_topic').value}\n"
            f"  joy  : {self.get_parameter('joy_topic').value}\n"
            f"  ctrl_mode (out): {self.get_parameter('control_mode_topic').value}"
        )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_leg_status(self, msg: MotorStatusMsg) -> None:
        now = self._now()
        with self._lock:
            self._leg_recv_count += 1
            self._leg_last_stamp = now
            for status in getattr(msg, "status", []):
                # status.name is the CAN ID, not a policy index
                can_id = int(getattr(status, "name", -1))
                idx = self._joint_map.index_by_can_id.get(can_id)
                if idx is None:
                    continue
                self._leg_pos[idx]    = float(getattr(status, "pos",     0.0))
                self._leg_vel[idx]    = float(getattr(status, "speed",   0.0))
                self._leg_torque[idx] = float(getattr(status, "current", 0.0))

    def _on_arm_status(self, msg: MotorStatusMsg) -> None:
        now = self._now()
        with self._lock:
            self._arm_recv_count += 1
            self._arm_last_stamp = now
            for status in getattr(msg, "status", []):
                can_id = int(getattr(status, "name", -1))
                idx = self._joint_map.index_by_can_id.get(can_id)
                if idx is None:
                    continue
                self._leg_pos[idx]    = float(getattr(status, "pos",     0.0))
                self._leg_vel[idx]    = float(getattr(status, "speed",   0.0))
                self._leg_torque[idx] = float(getattr(status, "current", 0.0))

    def _on_imu(self, msg: BodyImu) -> None:
        now = self._now()
        with self._lock:
            self._imu_recv_count += 1
            self._imu_last_stamp = now
            euler = getattr(msg, "euler", None)
            ang_vel = getattr(msg, "angular_velocity", None)
            if euler is not None:
                self._rpy[:] = [float(euler.roll), float(euler.pitch), float(euler.yaw)]
            if ang_vel is not None:
                self._ang_vel[:] = [float(ang_vel.x), float(ang_vel.y), float(ang_vel.z)]

    def _on_motion_reference(self, msg: MotionReference) -> None:
        now = self._now()
        with self._lock:
            self._motion_recv_count += 1
            self._motion_last_stamp = now
            body = np.asarray(msg.body_mimic, dtype=np.float32)
            if body.shape[0] == self._contract.n_mimic_obs:
                self._motion_ref = body

    def _on_joy(self, msg: Joy) -> None:
        now = self._now()
        with self._lock:
            self._joy_recv_count += 1
            self._joy_last_stamp = now
            self._joy_axes = list(getattr(msg, "axes", []))
            self._joy_buttons = list(getattr(msg, "buttons", []))

    def _on_control_mode(self, msg: ControlMode) -> None:
        with self._lock:
            self._mode_recv_count += 1
            self._latest_mode = int(msg.mode)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #

    _MODE_NAMES = {0: "STOP", 1: "ZERO", 2: "POLICY"}

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _age(self, stamp: float) -> str:
        if stamp == 0.0:
            return "never"
        age = self._now() - stamp
        return f"{age:.3f}s ago"

    def _print_summary(self) -> None:
        with self._lock:
            leg_count = self._leg_recv_count
            leg_age = self._age(self._leg_last_stamp)
            leg_pos = self._leg_pos.copy()
            leg_vel = self._leg_vel.copy()

            arm_count = self._arm_recv_count
            arm_age = self._age(self._arm_last_stamp)

            imu_count = self._imu_recv_count
            imu_age = self._age(self._imu_last_stamp)
            rpy = self._rpy.copy()
            ang_vel = self._ang_vel.copy()

            mot_count = self._motion_recv_count
            mot_age = self._age(self._motion_last_stamp)
            motion_ref = self._motion_ref.copy()

            joy_count = self._joy_recv_count
            joy_age = self._age(self._joy_last_stamp)
            joy_axes = list(self._joy_axes)
            joy_buttons = list(self._joy_buttons)

            mode_count = self._mode_recv_count
            latest_mode = self._latest_mode

        mode_str = self._MODE_NAMES.get(latest_mode, "?") if latest_mode is not None else "N/A"

        # ── compute obs_proprio (same formula as obs_builder)
        default_dof = self._contract.default_dof_pos
        ankle_idx   = self._contract.ankle_indices
        dof_vel_masked = leg_vel.copy()
        dof_vel_masked[ankle_idx] = 0.0
        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            leg_pos - default_dof,
            dof_vel_masked * 0.05,
            np.zeros(self._contract.num_actions, dtype=np.float32),  # last_action (not tracked here)
        ])
        obs_full = np.concatenate([motion_ref, obs_proprio]) if motion_ref.shape[0] == self._contract.n_mimic_obs else obs_proprio
        state_body = np.concatenate([ang_vel, rpy[:2], leg_pos])

        # ── header
        lines: list[str] = [
            "",
            _SEP,
            f"  [LEG  status] msgs={leg_count:6d}  last={leg_age}",
            f"  [ARM  status] msgs={arm_count:6d}  last={arm_age}",
            f"  [IMU  status] msgs={imu_count:6d}  last={imu_age}",
            f"  [MotionRef  ] msgs={mot_count:6d}  last={mot_age}",
            f"  [Joy/SBUS   ] msgs={joy_count:6d}  last={joy_age}",
            f"  [CtrlMode(out)] msgs={mode_count:5d}  mode={mode_str}",
            _SEP,
            "",
        ]

        # ── Section 1: IMU
        lines += [
            "\u2500\u2500 1. IMU / Body State " + "\u2500" * 50,
            f"  Quaternion (w,x,y,z) : (not available from bodyctrl IMU)",
            f"  RPY          (rad)   : roll={rpy[0]:+.4f}  pitch={rpy[1]:+.4f}  yaw={rpy[2]:+.4f}",
            f"  Angular vel  (raw)   : wx={ang_vel[0]:+.4f}  wy={ang_vel[1]:+.4f}  wz={ang_vel[2]:+.4f}",
            f"  Angular vel  \u00d70.25   : wx={ang_vel[0]*0.25:+.4f}  wy={ang_vel[1]*0.25:+.4f}  wz={ang_vel[2]*0.25:+.4f}",
            "",
        ]

        # ── Section 2: Joint State
        lines += ["\u2500\u2500 2. Joint State " + "\u2500" * 55]
        lines += _fmt_joint_table(leg_pos,                "dof_pos  (raw)")
        lines += _fmt_joint_table(leg_pos - default_dof,  "dof_pos - default")
        lines += _fmt_joint_table(leg_vel,                "dof_vel  (raw)")
        lines += _fmt_joint_table(dof_vel_masked * 0.05,  "dof_vel_masked \u00d70.05")
        lines += _fmt_joint_table(
            np.zeros(self._contract.num_actions, dtype=np.float32),
            "last_action"
        )
        lines.append("")

        # ── Section 3: Motion Reference
        lines += ["\u2500\u2500 3. Motion Reference (action_mimic, 26-dim) " + "\u2500" * 27]
        if motion_ref.shape[0] == self._contract.n_mimic_obs:
            lines += [
                f"  xy_vel_ref   : vx={motion_ref[0]:+.4f}  vy={motion_ref[1]:+.4f}",
                f"  z_pos_ref    : {motion_ref[2]:+.4f}",
                f"  roll_ref     : {motion_ref[3]:+.4f}    pitch_ref : {motion_ref[4]:+.4f}",
                f"  yaw_vel_ref  : {motion_ref[5]:+.4f}",
            ]
            lines += _fmt_joint_table(motion_ref[6:], "dof_ref  (mimic target, 20-dim)")
        else:
            lines.append(f"  (dim={motion_ref.shape[0]}, unexpected)")
        lines.append("")

        # ── Section 4: obs_proprio
        lines += [
            "\u2500\u2500 4. obs_proprio layout (65-dim) " + "\u2500" * 39,
            f"  [0: 3]  ang_vel\u00d70.25    : {_fmt_vec(obs_proprio[0:3])}",
            f"  [3: 5]  rpy[:2]         : {_fmt_vec(obs_proprio[3:5])}",
            f"  [5:25]  dof_pos-default : {_fmt_vec(obs_proprio[5:25], '+.3f')}",
            f"  [25:45] dof_vel_m\u00d70.05  : {_fmt_vec(obs_proprio[25:45], '+.3f')}",
            f"  [45:65] last_action     : {_fmt_vec(obs_proprio[45:65], '+.3f')}",
            "",
        ]

        # ── Section 5: obs_full
        lines += [
            "\u2500\u2500 5. obs_full layout (91-dim = mimic26 + proprio65) " + "\u2500" * 19,
            "  obs_full[0:26]  = action_mimic",
            "  obs_full[26:91] = obs_proprio",
            f"  L2 norm: {np.linalg.norm(obs_full):.4f}",
            "",
        ]

        # ── Section 6: state_body
        lines += [
            "\u2500\u2500 6. state_body (25-dim) " + "\u2500" * 47,
            f"  [0:3]  ang_vel  : {_fmt_vec(state_body[0:3])}",
            f"  [3:5]  rpy[:2]  : {_fmt_vec(state_body[3:5])}",
            f"  [5:25] dof_pos  : {_fmt_vec(state_body[5:25], '+.3f')}",
            "",
        ]

        # ── verbose: Joy
        if self._verbose:
            lines += [
                "\u2500\u2500 7. Joystick (SBUS) " + "\u2500" * 51,
                f"  Joy axes : {[f'{a:.3f}' for a in joy_axes]}",
                f"  Joy btns : {joy_buttons}",
                "",
            ]

        lines.append(_SEP)

        # Print as a single block so concurrent output doesn't interleave
        print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InputMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
