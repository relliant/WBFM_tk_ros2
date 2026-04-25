from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from tienkung_interfaces.msg import ControlMode, MotionReference

from .motion_lib_adapter import DEFAULT_MIMIC_OBS_TIENKUNG, MotionLibAdapter, build_mimic_obs


class MotionSourceNode(Node):
    def __init__(self) -> None:
        super().__init__("tienkung_motion_source")
        self.declare_parameter("motion_file", "")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("control_dt", 0.02)
        self.declare_parameter("start_paused", True)
        self.declare_parameter("loop", True)
        self.declare_parameter("send_start_frame_as_end_frame", False)
        self.declare_parameter("future_steps", [1])
        self.declare_parameter("control_mode_topic", "/tienkung/control_mode")
        self.declare_parameter("motion_reference_topic", "/tienkung/motion_reference")

        self.control_dt = float(self.get_parameter("control_dt").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.future_steps = [int(step) for step in self.get_parameter("future_steps").value]
        self.loop = bool(self.get_parameter("loop").value)
        self.start_paused = bool(self.get_parameter("start_paused").value)
        self.send_start_frame_as_end_frame = bool(self.get_parameter("send_start_frame_as_end_frame").value)
        motion_file = str(self.get_parameter("motion_file").value)
        self.motion_lib: Optional[MotionLibAdapter] = MotionLibAdapter(motion_file) if motion_file else None

        self.current_mode = ControlMode.STOP if self.start_paused else ControlMode.POLICY
        self.t_step = 0
        self.motion_reference_pub = self.create_publisher(MotionReference, str(self.get_parameter("motion_reference_topic").value), 10)
        self.create_subscription(ControlMode, str(self.get_parameter("control_mode_topic").value), self._control_mode_callback, 10)
        self.timer = self.create_timer(1.0 / self.publish_hz, self._publish_reference)

        self.default_body = DEFAULT_MIMIC_OBS_TIENKUNG.astype(np.float32)
        self.start_frame_body = self._compute_start_frame() if self.motion_lib is not None else self.default_body

    def _compute_start_frame(self) -> np.ndarray:
        if self.motion_lib is None:
            return self.default_body
        return build_mimic_obs(self.motion_lib, 0, self.control_dt, self.future_steps)

    def _control_mode_callback(self, msg: ControlMode) -> None:
        self.current_mode = msg.mode

    def _idle_body(self) -> np.ndarray:
        if self.send_start_frame_as_end_frame:
            return self.start_frame_body
        return self.default_body

    def _publish_reference(self) -> None:
        msg = MotionReference()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.left_hand = []
        msg.right_hand = []
        msg.neck = []

        if self.motion_lib is None or self.current_mode != ControlMode.POLICY:
            body = self._idle_body()
        else:
            body = build_mimic_obs(self.motion_lib, self.t_step, self.control_dt, self.future_steps)
            self.t_step += 1
            max_steps = max(1, int(self.motion_lib.get_motion_length() / self.control_dt))
            if self.t_step >= max_steps:
                self.t_step = 0 if self.loop else max_steps - 1

        msg.body_mimic = np.asarray(body, dtype=np.float32).tolist()
        self.motion_reference_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MotionSourceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
