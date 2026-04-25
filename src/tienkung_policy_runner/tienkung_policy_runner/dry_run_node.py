"""dry_run_node.py

安全验证节点：
  STOP   模式 — 不发任何电机指令
  ZERO   模式 — 发真实 PD 位置指令，目标为 default_dof_pos（归零位）
  POLICY 模式 — 只推理，不发指令（dry run）

操作流程
--------
1. 启动节点，机器人处于 STOP（不动）
2. 拨动遥控器到 ZERO → 机器人缓慢运动到默认站姿
3. 待归零完成后拨动到 POLICY → 开始 dry-run，观察推理输出是否合理
4. 确认输出正常后，再切换到正式 policy_runner_node

每 N 秒打印：推理帧率/延迟、topic 状态、body state、obs 统计、action 输出
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any, Deque, Optional

import numpy as np
import rclpy
from bodyctrl_msgs.msg import CmdMotorCtrl, CmdSetMotorPosition
from bodyctrl_msgs.msg import Imu as BodyImu
from bodyctrl_msgs.msg import MotorStatusMsg
from rclpy.node import Node
from sensor_msgs.msg import Joy

from tienkung_interfaces.msg import MotionReference

from .fsm import ControlMode, JoystickCommand, PolicyFSM, decode_joy_message
from .manifest_loader import load_manifest, validate_manifest_against_contract
from .obs_builder import TienkungObservationBuilder, postprocess_action
from .onnx_runtime import OnnxPolicyRunner, inspect_policy
from .robot_contract import DEFAULT_MIMIC_OBS_TIENKUNG, get_robot_contract
from .robot_io import JointMap, RobotIO, default_gains


# ─────────────────────────────────────────────────────────────────────────────
# Statistics accumulator
# ─────────────────────────────────────────────────────────────────────────────

class RollingStats:
    """保存最近 N 个样本的滚动统计量。"""

    def __init__(self, maxlen: int = 500) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        self._buf.append(value)

    def summary(self) -> dict:
        if not self._buf:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        arr = np.asarray(self._buf, dtype=np.float64)
        return {
            "n":    len(arr),
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
        }

    def clear(self) -> None:
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
# DryRunNode
# ─────────────────────────────────────────────────────────────────────────────

class DryRunNode(Node):
    """订阅所有输入 topic，执行推理但不发布任何电机指令。"""

    # ── 格式化辅助 ─────────────────────────────────────────────────────────

    _JOINT_NAMES = [
        "hip_roll_l",   "hip_pitch_l",  "hip_yaw_l",    "knee_l",
        "ankle_p_l",    "ankle_r_l",
        "hip_roll_r",   "hip_pitch_r",  "hip_yaw_r",    "knee_r",
        "ankle_p_r",    "ankle_r_r",
        "shld_p_l",     "shld_r_l",     "shld_y_l",     "elbow_l",
        "shld_p_r",     "shld_r_r",     "shld_y_r",     "elbow_r",
    ]

    # ── Initialisation ────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__("tienkung_dry_run")

        # ---------- parameters ----------
        self.declare_parameter("device",               "cpu")
        self.declare_parameter("policy_path",          "")
        self.declare_parameter("manifest_path",        "")
        self.declare_parameter("policy_hz",            50.0)
        self.declare_parameter("report_interval_sec",  5.0)
        self.declare_parameter("state_timeout_sec",    0.1)
        self.declare_parameter("motion_timeout_sec",   0.1)
        self.declare_parameter("zero_duration_sec",    2.0)
        self.declare_parameter("motion_reference_topic", "/tienkung/motion_reference")
        self.declare_parameter("leg_status_topic",     "/leg/status")
        self.declare_parameter("arm_status_topic",     "/arm/status")
        self.declare_parameter("imu_status_topic",     "/imu/status")
        self.declare_parameter("joy_topic",            "/sbus_data")
        self.declare_parameter("leg_command_topic",    "/leg/cmd_ctrl")
        self.declare_parameter("arm_command_topic",    "/arm/cmd_ctrl")
        self.declare_parameter("waist_command_topic",  "/waist/cmd_pos")

        self._contract = get_robot_contract("tienkung")

        # ---------- manifest (optional) ----------
        manifest_path = str(self.get_parameter("manifest_path").value)
        if manifest_path:
            manifest = load_manifest(manifest_path)
            validate_manifest_against_contract(manifest, self._contract)

        # ---------- policy ----------
        policy_path = str(self.get_parameter("policy_path").value)
        if not policy_path:
            raise ValueError("[DryRun] policy_path parameter is required")

        info = inspect_policy(policy_path)
        if info:
            self.get_logger().info(
                f"[DryRun] ONNX model info:\n"
                f"  input : {info['input_name']} {info['input_shape']}\n"
                f"  output: {info['output_name']} {info['output_shape']}\n"
                f"  providers: {info['providers']}"
            )
        self._policy = OnnxPolicyRunner(
            policy_path, str(self.get_parameter("device").value)
        )

        # ---------- IO helpers ----------
        from pathlib import Path
        joint_map_path = Path(__file__).resolve().parents[1] / "config" / "joint_map.yaml"
        if not joint_map_path.exists():
            joint_map_path = Path(__file__).resolve().parents[2] / "config" / "joint_map.yaml"
        self._joint_map = JointMap.from_yaml(str(joint_map_path))
        self._robot_io  = RobotIO(self._joint_map, self._contract)
        self._obs_builder = TienkungObservationBuilder(self._contract)
        self._fsm = PolicyFSM(
            zero_duration_sec=float(self.get_parameter("zero_duration_sec").value)
        )
        self._kp, self._kd = default_gains(self._contract)
        self._zero_traj_duration_sec = max(
            1e-3, float(self.get_parameter("zero_duration_sec").value)
        )
        self._zero_start_time_sec: Optional[float] = None
        self._zero_start_dof_pos = self._contract.default_dof_pos.copy()
        self._prev_mode = ControlMode.STOP

        # ---------- state ----------
        self._lock = threading.Lock()
        self._last_action = np.zeros(self._contract.num_actions, dtype=np.float32)
        self._latest_motion_ref = DEFAULT_MIMIC_OBS_TIENKUNG.copy()
        self._last_motion_ref_time: float = 0.0

        self._state_timeout  = float(self.get_parameter("state_timeout_sec").value)
        self._motion_timeout = float(self.get_parameter("motion_timeout_sec").value)
        self._report_interval = float(self.get_parameter("report_interval_sec").value)

        # ---------- statistics ----------
        self._infer_latency_ms = RollingStats(maxlen=2000)   # ms per step
        self._infer_interval_ms = RollingStats(maxlen=2000)  # inter-step ms → FPS
        self._last_infer_time: Optional[float] = None

        self._tick_count   = 0
        self._active_count = 0   # ticks where mode == POLICY
        self._last_report_time: float = time.monotonic()

        # Topic receive counters
        self._recv = {k: 0 for k in ("leg", "arm", "imu", "motion", "joy")}
        self._last_stamp: dict[str, float] = {k: 0.0 for k in self._recv}

        # ---------- subscriptions ----------
        self.create_subscription(
            MotionReference,
            str(self.get_parameter("motion_reference_topic").value),
            self._cb_motion_reference, 10,
        )
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("leg_status_topic").value),
            self._cb_leg_status, 100,
        )
        self.create_subscription(
            MotorStatusMsg,
            str(self.get_parameter("arm_status_topic").value),
            self._cb_arm_status, 100,
        )
        self.create_subscription(
            BodyImu,
            str(self.get_parameter("imu_status_topic").value),
            self._cb_imu, 100,
        )
        self.create_subscription(
            Joy,
            str(self.get_parameter("joy_topic").value),
            self._cb_joy, 100,
        )

        # ---------- publishers (ZERO mode only) ----------
        self._leg_cmd_pub = self.create_publisher(
            CmdMotorCtrl,
            str(self.get_parameter("leg_command_topic").value), 10,
        )
        self._arm_cmd_pub = self.create_publisher(
            CmdMotorCtrl,
            str(self.get_parameter("arm_command_topic").value), 10,
        )
        self._waist_cmd_pub = self.create_publisher(
            CmdSetMotorPosition,
            str(self.get_parameter("waist_command_topic").value), 10,
        )

        # ---------- timer ----------
        hz = float(self.get_parameter("policy_hz").value)
        self._timer = self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            "[DryRun] Node started.\n"
            "  STOP   → no motor output\n"
            "  ZERO   → sends default_dof_pos PD commands (real motor output!)\n"
            "  POLICY → inference only, no motor output (dry run)\n"
            f"  policy_hz      = {hz}\n"
            f"  zero_duration  = {self.get_parameter('zero_duration_sec').value}s "
            "(ramp+gating)\n"
            f"  report_every   = {self._report_interval}s\n"
            f"  leg cmd topic  = {self.get_parameter('leg_command_topic').value}\n"
            f"  arm cmd topic  = {self.get_parameter('arm_command_topic').value}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_motion_reference(self, msg: MotionReference) -> None:
        body = np.asarray(msg.body_mimic, dtype=np.float32)
        now = self._now()
        with self._lock:
            self._recv["motion"] += 1
            self._last_stamp["motion"] = now
            if body.shape[0] == self._contract.n_mimic_obs:
                self._latest_motion_ref = body
                self._last_motion_ref_time = now

    def _cb_leg_status(self, msg: MotorStatusMsg) -> None:
        now = self._now()
        with self._lock:
            self._recv["leg"] += 1
            self._last_stamp["leg"] = now
        self._robot_io.ingest_leg_status(msg, now)

    def _cb_arm_status(self, msg: MotorStatusMsg) -> None:
        now = self._now()
        with self._lock:
            self._recv["arm"] += 1
            self._last_stamp["arm"] = now
        self._robot_io.ingest_arm_status(msg, now)

    def _cb_imu(self, msg: BodyImu) -> None:
        now = self._now()
        with self._lock:
            self._recv["imu"] += 1
            self._last_stamp["imu"] = now
        self._robot_io.ingest_imu(msg, now)

    def _cb_joy(self, msg: Joy) -> None:
        now = self._now()
        with self._lock:
            self._recv["joy"] += 1
            self._last_stamp["joy"] = now
        self._robot_io.ingest_joy(msg)

    # ── Main tick ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = self._now()
        state = self._robot_io.snapshot()
        joy   = self._robot_io.last_joy

        # FSM 状态判断（不允许 POLICY 发出指令，但需要知道当前逻辑状态）
        with self._lock:
            motion_ref  = self._latest_motion_ref.copy()
            motion_time = self._last_motion_ref_time

        state_ready  = (now - state.timestamp_sec)  <= self._state_timeout  and state.timestamp_sec  > 0.0
        motion_ready = (now - motion_time)           <= self._motion_timeout and motion_time           > 0.0

        if joy is None:
            joy = JoystickCommand(requested_mode=None, disable=False)

        mode = self._fsm.update(now, joy, state_ready, motion_ready, True)

        if mode == ControlMode.ZERO and self._prev_mode != ControlMode.ZERO:
            self._zero_start_time_sec = now
            self._zero_start_dof_pos = state.dof_pos.copy()
        elif mode != ControlMode.ZERO:
            self._zero_start_time_sec = None

        self._tick_count += 1

        # ── ZERO 模式：发真实 PD 归零指令
        if mode == ControlMode.ZERO:
            if self._zero_start_time_sec is None:
                self._zero_start_time_sec = now
                self._zero_start_dof_pos = state.dof_pos.copy()

            alpha = np.clip(
                (now - self._zero_start_time_sec) / self._zero_traj_duration_sec,
                0.0,
                1.0,
            )
            blend = self._quintic_blend(float(alpha))
            target = (
                self._zero_start_dof_pos
                + (self._contract.default_dof_pos - self._zero_start_dof_pos) * blend
            ).astype(np.float32)
            cmd_dict = self._robot_io.build_command_dict(target, self._kp, self._kd)
            leg_msg, arm_msg, waist_msg = self._robot_io.build_ros_messages(self, cmd_dict)
            self._leg_cmd_pub.publish(leg_msg)
            self._arm_cmd_pub.publish(arm_msg)
            self._waist_cmd_pub.publish(waist_msg)

        # ── 推理（所有模式都执行，用于测量帧率；POLICY 模式不发指令）
        obs = self._obs_builder.build_observation(
            motion_ref,
            state.ang_vel,
            state.rpy,
            state.dof_pos,
            state.dof_vel,
            self._last_action,
        )

        t0 = time.monotonic()
        raw_action = self._policy.infer(obs)[0]
        t1 = time.monotonic()

        latency_ms = (t1 - t0) * 1e3
        self._infer_latency_ms.push(latency_ms)

        if self._last_infer_time is not None:
            interval_ms = (t1 - self._last_infer_time) * 1e3
            self._infer_interval_ms.push(interval_ms)
        self._last_infer_time = t1

        # POLICY 模式下才更新 last_action（避免 ZERO 期间累积发散）
        if mode == ControlMode.POLICY:
            self._last_action = np.asarray(raw_action, dtype=np.float32)
            self._active_count += 1
        elif mode != ControlMode.POLICY:
            # STOP/ZERO 期间重置 last_action 到零，防止冷启动发散
            self._last_action = np.zeros(self._contract.num_actions, dtype=np.float32)

        self._prev_mode = mode

        # ── 定期打印报告
        elapsed = time.monotonic() - self._last_report_time
        if elapsed >= self._report_interval:
            self._print_report(now, state, motion_ref, obs, raw_action, mode)
            self._last_report_time = time.monotonic()

    # ── Report printer ───────────────────────────────────────────────────

    _MODE_NAMES = {
        ControlMode.STOP:   "STOP  ",
        ControlMode.ZERO:   "ZERO  ",
        ControlMode.POLICY: "POLICY",
    }

    def _age(self, stamp: float, now: float) -> str:
        if stamp == 0.0:
            return "never"
        return f"{now - stamp:.3f}s ago"

    @staticmethod
    def _quintic_blend(alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        # 与 ros_lite 的 FifthPoly(零速度/零加速度边界)等价的归一化插值
        return alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)

    def _obs_segment_stats(self, obs: np.ndarray) -> list[str]:
        """将 obs 各段统计打印出来。"""
        n_mimic   = self._contract.n_mimic_obs    # 26
        n_proprio = self._contract.n_proprio       # 65
        n_single  = self._contract.n_obs_single    # 91
        hist_len  = self._contract.history_len     # 10

        segments = [
            ("obs[ 0:26]  action_mimic",    obs[0:26]),
            ("obs[26:29]  ang_vel×0.25",    obs[26:29]),
            ("obs[29:31]  rpy[:2]",         obs[29:31]),
            ("obs[31:51]  dof_pos-default", obs[31:51]),
            ("obs[51:71]  dof_vel×0.05",    obs[51:71]),
            ("obs[71:91]  last_action",     obs[71:91]),
        ]
        lines = ["  obs_full segment stats (current frame):"]
        for name, seg in segments:
            if seg.size == 0:
                continue
            lines.append(
                f"    {name:38s}  "
                f"mean={np.mean(seg):+.4f}  std={np.std(seg):.4f}  "
                f"max_abs={np.max(np.abs(seg)):.4f}"
            )
        return lines

    def _action_stats(self, raw_action: np.ndarray) -> list[str]:
        processed = postprocess_action(raw_action, self._contract)
        lines = ["  raw_action / processed_action per joint:"]
        header = f"    {'joint':>14s}  {'raw':>8s}  {'processed':>10s}"
        lines.append(header)
        for i, name in enumerate(self._JOINT_NAMES):
            lines.append(
                f"    {name:>14s}  {raw_action[i]:>+8.4f}  {processed[i]:>+10.4f}"
            )
        return lines

    def _print_report(
        self,
        now: float,
        state: Any,
        motion_ref: np.ndarray,
        obs: np.ndarray,
        raw_action: np.ndarray,
        mode: ControlMode,
    ) -> None:
        lat  = self._infer_latency_ms.summary()
        intv = self._infer_interval_ms.summary()

        # FPS from interval
        fps_mean = 1000.0 / intv["mean"]  if intv["mean"] > 0 else float("nan")
        fps_min  = 1000.0 / intv["max"]   if intv["max"]  > 0 else float("nan")
        fps_max  = 1000.0 / intv["min"]   if intv["min"]  > 0 else float("nan")

        sep  = "=" * 72
        sep2 = "-" * 72

        with self._lock:
            recv  = dict(self._recv)
            stamp = dict(self._last_stamp)

        mode_banner = {
            ControlMode.STOP:   "STOP   — no motor output",
            ControlMode.ZERO:   "ZERO   — sending default_dof_pos commands  ⚠ REAL MOTOR OUTPUT",
            ControlMode.POLICY: "POLICY — inference only, NO motor output  (dry run)",
        }[mode]

        lines: list[str] = [
            "",
            sep,
            f"[DryRun]  {mode_banner}",
            sep,
            f"  FSM mode        : {self._MODE_NAMES[mode]}",
            f"  total ticks     : {self._tick_count}   "
            f"active (POLICY) ticks: {self._active_count}",
            "",
            sep2,
            "  INFERENCE PERFORMANCE",
            sep2,
            f"  Latency  (ms)   : mean={lat['mean']:6.2f}  std={lat['std']:5.2f}"
            f"  min={lat['min']:5.2f}  max={lat['max']:6.2f}  (n={lat['n']})",
            f"  Interval (ms)   : mean={intv['mean']:6.2f}  std={intv['std']:5.2f}"
            f"  min={intv['min']:5.2f}  max={intv['max']:6.2f}",
            f"  FPS   (Hz)      : mean={fps_mean:6.2f}  min={fps_min:5.2f}  max={fps_max:6.2f}",
            f"  Target FPS      : {self.get_parameter('policy_hz').value:.1f} Hz",
            "",
            sep2,
            "  TOPIC STATUS",
            sep2,
            f"  leg  /leg/status         : recv={recv['leg']:6d}  last={self._age(stamp['leg'],  now)}",
            f"  arm  /arm/status         : recv={recv['arm']:6d}  last={self._age(stamp['arm'],  now)}",
            f"  imu  /imu/status         : recv={recv['imu']:6d}  last={self._age(stamp['imu'],  now)}",
            f"  motion_reference         : recv={recv['motion']:6d}  last={self._age(stamp['motion'],now)}",
            f"  joy  /sbus_data          : recv={recv['joy']:6d}  last={self._age(stamp['joy'],  now)}",
            "",
            sep2,
            "  BODY STATE",
            sep2,
            f"  rpy   (rad)  : roll={state.rpy[0]:+.4f}  pitch={state.rpy[1]:+.4f}  yaw={state.rpy[2]:+.4f}",
            f"  ang_vel      : wx={state.ang_vel[0]:+.4f}  wy={state.ang_vel[1]:+.4f}  wz={state.ang_vel[2]:+.4f}",
            "",
            sep2,
            "  MOTION REFERENCE (action_mimic, 26-dim)",
            sep2,
            f"  xy_vel_ref   : vx={motion_ref[0]:+.4f}  vy={motion_ref[1]:+.4f}",
            f"  z_pos_ref    : {motion_ref[2]:+.4f}",
            f"  rp_ref       : roll={motion_ref[3]:+.4f}  pitch={motion_ref[4]:+.4f}",
            f"  yaw_vel_ref  : {motion_ref[5]:+.4f}",
            f"  dof_ref      : [{', '.join(f'{v:+.3f}' for v in motion_ref[6:])}]",
            "",
            sep2,
            "  OBSERVATION STATS",
            sep2,
        ]
        lines += self._obs_segment_stats(obs)
        lines += [
            f"  total obs norm : {np.linalg.norm(obs):.4f}  "
            f"(total_obs_size={self._contract.total_obs_size})",
            "",
            sep2,
            "  RAW ACTION OUTPUT",
            sep2,
        ]
        lines += self._action_stats(raw_action)
        lines += [
            f"\n  raw_action  norm : {np.linalg.norm(raw_action):.4f}  "
            f"max_abs={np.max(np.abs(raw_action)):.4f}",
            sep,
        ]

        # Reset rolling stats after each report (show per-interval stats)
        self._infer_latency_ms.clear()
        self._infer_interval_ms.clear()

        print("\n".join(lines), flush=True)


# ─────────────────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DryRunNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
