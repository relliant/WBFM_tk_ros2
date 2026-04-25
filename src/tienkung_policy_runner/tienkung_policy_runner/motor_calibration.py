from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"Expected {name} shape {(size,)}, got {array.shape}")
    return array


@dataclass
class MotorCalibration:
    zero_pos_offset: np.ndarray
    zero_offset: np.ndarray
    motor_dir: np.ndarray
    ct_scale: np.ndarray
    wrap_threshold_rad: float = float(np.pi)
    enable_jump_filter: bool = True

    def __post_init__(self) -> None:
        size = int(self.zero_pos_offset.shape[0])
        self.zero_pos_offset = _as_vector(self.zero_pos_offset, size, "zero_pos_offset")
        self.zero_offset = _as_vector(self.zero_offset, size, "zero_offset")
        self.motor_dir = _as_vector(self.motor_dir, size, "motor_dir")
        self.ct_scale = _as_vector(self.ct_scale, size, "ct_scale")
        self.wrap_threshold_rad = float(self.wrap_threshold_rad)
        self.enable_jump_filter = bool(self.enable_jump_filter)

    @classmethod
    def from_dict(cls, data: dict, num_motors: int) -> "MotorCalibration":
        return cls(
            zero_pos_offset=_as_vector(
                data.get("zero_pos_offset", np.zeros(num_motors, dtype=np.float32)),
                num_motors,
                "zero_pos_offset",
            ),
            zero_offset=_as_vector(
                data.get("zero_offset", np.zeros(num_motors, dtype=np.float32)),
                num_motors,
                "zero_offset",
            ),
            motor_dir=_as_vector(
                data.get("motor_dir", np.ones(num_motors, dtype=np.float32)),
                num_motors,
                "motor_dir",
            ),
            ct_scale=_as_vector(
                data.get("ct_scale", np.zeros(num_motors, dtype=np.float32)),
                num_motors,
                "ct_scale",
            ),
            wrap_threshold_rad=float(data.get("wrap_threshold_rad", np.pi)),
            enable_jump_filter=bool(data.get("enable_jump_filter", True)),
        )

    def convert_feedback(
        self,
        raw_pos: Sequence[float],
        raw_vel: Sequence[float],
        raw_current: Sequence[float],
        zero_cnt: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_pos = _as_vector(raw_pos, self.zero_pos_offset.shape[0], "raw_pos")
        raw_vel = _as_vector(raw_vel, self.zero_pos_offset.shape[0], "raw_vel")
        raw_current = _as_vector(raw_current, self.zero_pos_offset.shape[0], "raw_current")
        zero_cnt = _as_vector(zero_cnt, self.zero_pos_offset.shape[0], "zero_cnt")

        torque = raw_current * self.ct_scale
        q = (raw_pos - self.zero_pos_offset) * self.motor_dir + self.zero_offset
        zero_cnt = np.where(q > self.wrap_threshold_rad, -1.0, zero_cnt)
        zero_cnt = np.where(q < -self.wrap_threshold_rad, 1.0, zero_cnt)
        q = q + zero_cnt * (2.0 * np.pi)
        qdot = raw_vel * self.motor_dir
        tau = torque * self.motor_dir
        return q.astype(np.float32), qdot.astype(np.float32), tau.astype(np.float32), zero_cnt.astype(np.float32)

    def convert_command(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        joint_torque: Sequence[float],
        zero_cnt: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_pos = _as_vector(joint_pos, self.zero_pos_offset.shape[0], "joint_pos")
        joint_vel = _as_vector(joint_vel, self.zero_pos_offset.shape[0], "joint_vel")
        joint_torque = _as_vector(joint_torque, self.zero_pos_offset.shape[0], "joint_torque")
        zero_cnt = _as_vector(zero_cnt, self.zero_pos_offset.shape[0], "zero_cnt")

        motor_pos = (
            (joint_pos - self.zero_offset - zero_cnt * (2.0 * np.pi)) * self.motor_dir
            + self.zero_pos_offset
        )
        motor_vel = joint_vel * self.motor_dir
        motor_torque = joint_torque * self.motor_dir
        return motor_pos.astype(np.float32), motor_vel.astype(np.float32), motor_torque.astype(np.float32)

    def reject_large_position_jumps(
        self,
        raw_pos: Sequence[float],
        raw_vel: Sequence[float],
        raw_current: Sequence[float],
        last_raw_pos: Sequence[float],
        last_raw_vel: Sequence[float],
        last_raw_current: Sequence[float],
        valid_mask: Sequence[bool],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_pos = _as_vector(raw_pos, self.zero_pos_offset.shape[0], "raw_pos")
        raw_vel = _as_vector(raw_vel, self.zero_pos_offset.shape[0], "raw_vel")
        raw_current = _as_vector(raw_current, self.zero_pos_offset.shape[0], "raw_current")
        last_raw_pos = _as_vector(last_raw_pos, self.zero_pos_offset.shape[0], "last_raw_pos")
        last_raw_vel = _as_vector(last_raw_vel, self.zero_pos_offset.shape[0], "last_raw_vel")
        last_raw_current = _as_vector(last_raw_current, self.zero_pos_offset.shape[0], "last_raw_current")
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if not self.enable_jump_filter:
            return raw_pos, raw_vel, raw_current, valid_mask

        large_jump = valid_mask & (np.abs(raw_pos - last_raw_pos) > self.wrap_threshold_rad)
        filtered_pos = np.where(large_jump, last_raw_pos, raw_pos)
        filtered_vel = np.where(large_jump, last_raw_vel, raw_vel)
        filtered_current = np.where(large_jump, last_raw_current, raw_current)
        valid_mask = valid_mask | np.ones_like(valid_mask, dtype=bool)
        return (
            filtered_pos.astype(np.float32),
            filtered_vel.astype(np.float32),
            filtered_current.astype(np.float32),
            valid_mask,
        )
