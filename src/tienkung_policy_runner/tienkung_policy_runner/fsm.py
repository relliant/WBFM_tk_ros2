from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ControlMode(Enum):
    STOP = 0
    ZERO = 1
    POLICY = 2


@dataclass
class JoystickCommand:
    requested_mode: Optional[ControlMode]
    disable: bool
    x_speed_command: float = 0.0
    y_speed_command: float = 0.0
    yaw_speed_command: float = 0.0


class PolicyFSM:
    def __init__(self, zero_duration_sec: float = 2.0) -> None:
        self.mode = ControlMode.STOP
        self.zero_duration_sec = float(zero_duration_sec)
        self.zero_start_time_sec: Optional[float] = None

    def update(self, now_sec: float, joystick: JoystickCommand, state_ready: bool, motion_ready: bool, policy_ready: bool) -> ControlMode:
        if joystick.disable or not state_ready:
            self.mode = ControlMode.STOP
            self.zero_start_time_sec = None
            return self.mode

        if joystick.requested_mode == ControlMode.STOP:
            self.mode = ControlMode.STOP
            self.zero_start_time_sec = None
        elif joystick.requested_mode == ControlMode.ZERO:
            if self.mode != ControlMode.ZERO:
                self.zero_start_time_sec = now_sec
            self.mode = ControlMode.ZERO
        elif joystick.requested_mode == ControlMode.POLICY:
            zero_done = self.zero_start_time_sec is not None and (now_sec - self.zero_start_time_sec) >= self.zero_duration_sec
            if motion_ready and policy_ready and (self.mode == ControlMode.POLICY or zero_done):
                self.mode = ControlMode.POLICY
            elif self.mode != ControlMode.ZERO:
                self.mode = ControlMode.ZERO
                self.zero_start_time_sec = now_sec

        if self.mode == ControlMode.POLICY and (not motion_ready or not policy_ready):
            self.mode = ControlMode.ZERO if state_ready else ControlMode.STOP
        return self.mode


def decode_joy_message(axes: list[float], buttons: list[int]) -> JoystickCommand:
    requested_mode: Optional[ControlMode] = None
    disable = False
    x_speed_command = 0.0
    y_speed_command = 0.0
    yaw_speed_command = 0.0

    if len(axes) == 12:
        a, c, d, e, g = axes[8], axes[10], axes[11], axes[4], axes[5]
        x1, y1, y2 = axes[3], axes[2], axes[0]
        disable = e == 1.0 and axes[9] == 1.0
        if d == 1.0:
            requested_mode = ControlMode.ZERO
        elif c == 1.0:
            requested_mode = ControlMode.STOP
        elif a == 1.0 and g == 0.0:
            requested_mode = ControlMode.POLICY
        y_speed_command = x1 * -0.4
        x_speed_command = y1 * (0.8 if y1 >= 0 else 0.5)
        yaw_speed_command = y2 * -0.4
    elif len(axes) >= 5 and len(buttons) >= 8:
        disable = bool(buttons[4] and buttons[1])
        if buttons[2]:
            requested_mode = ControlMode.ZERO
        elif buttons[3]:
            requested_mode = ControlMode.STOP
        elif buttons[0] and not buttons[6]:
            requested_mode = ControlMode.POLICY
        y_speed_command = axes[0] * -0.4
        x_speed_command = axes[1] * (0.8 if axes[1] >= 0 else 0.5)
        yaw_speed_command = axes[4] * -0.4

    return JoystickCommand(
        requested_mode=requested_mode,
        disable=disable,
        x_speed_command=float(x_speed_command),
        y_speed_command=float(y_speed_command),
        yaw_speed_command=float(yaw_speed_command),
    )


def interpolate_zero_pose(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (current + (target - current) * alpha).astype(np.float32)
