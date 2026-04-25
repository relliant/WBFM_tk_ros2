from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class LinearJointTransmission:
    """Linear motor<->joint-space mapping for a coupled ankle pair.

    `motor_to_joint` maps the two physical ankle motor coordinates into the
    policy-facing ankle pitch/roll joint coordinates.
    """

    name: str
    motor_indices: tuple[int, int]
    joint_indices: tuple[int, int]
    motor_to_joint: np.ndarray

    def __post_init__(self) -> None:
        self.motor_indices = tuple(int(index) for index in self.motor_indices)
        self.joint_indices = tuple(int(index) for index in self.joint_indices)
        self.motor_to_joint = np.asarray(self.motor_to_joint, dtype=np.float32).reshape(2, 2)
        self.joint_to_motor = np.linalg.inv(self.motor_to_joint).astype(np.float32)
        self.motor_torque_to_joint = np.linalg.inv(self.motor_to_joint).T.astype(np.float32)
        self.joint_torque_to_motor = self.motor_to_joint.T.astype(np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "LinearJointTransmission":
        return cls(
            name=str(data["name"]),
            motor_indices=tuple(data["motor_indices"]),
            joint_indices=tuple(data["joint_indices"]),
            motor_to_joint=np.asarray(data["motor_to_joint"], dtype=np.float32),
        )

    def motor_state_to_joint_state(
        self,
        motor_pos: Sequence[float],
        motor_vel: Sequence[float],
        motor_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motor_pos = np.asarray(motor_pos, dtype=np.float32)
        motor_vel = np.asarray(motor_vel, dtype=np.float32)
        motor_torque = np.asarray(motor_torque, dtype=np.float32)
        return (
            (self.motor_to_joint @ motor_pos).astype(np.float32),
            (self.motor_to_joint @ motor_vel).astype(np.float32),
            (self.motor_torque_to_joint @ motor_torque).astype(np.float32),
        )

    def joint_command_to_motor_space(
        self,
        joint_pos: Sequence[float],
        joint_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_torque = np.asarray(joint_torque, dtype=np.float32)
        return (
            (self.joint_to_motor @ joint_pos).astype(np.float32),
            (self.joint_torque_to_motor @ joint_torque).astype(np.float32),
        )


def apply_motor_state_transmissions(
    dof_pos: Sequence[float],
    dof_vel: Sequence[float],
    dof_torque: Sequence[float],
    transmissions: Iterable[LinearJointTransmission],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos = np.asarray(dof_pos, dtype=np.float32).copy()
    joint_vel = np.asarray(dof_vel, dtype=np.float32).copy()
    joint_torque = np.asarray(dof_torque, dtype=np.float32).copy()
    raw_pos = joint_pos.copy()
    raw_vel = joint_vel.copy()
    raw_torque = joint_torque.copy()

    for transmission in transmissions:
        motor_indices = list(transmission.motor_indices)
        indices = list(transmission.joint_indices)
        pos_pair, vel_pair, torque_pair = transmission.motor_state_to_joint_state(
            raw_pos[motor_indices],
            raw_vel[motor_indices],
            raw_torque[motor_indices],
        )
        joint_pos[indices] = pos_pair
        joint_vel[indices] = vel_pair
        joint_torque[indices] = torque_pair

    return joint_pos, joint_vel, joint_torque


def apply_joint_command_transmissions(
    target_dof_pos: Sequence[float],
    torques: Sequence[float],
    transmissions: Iterable[LinearJointTransmission],
) -> tuple[np.ndarray, np.ndarray]:
    joint_target = np.asarray(target_dof_pos, dtype=np.float32)
    joint_torques = np.asarray(torques, dtype=np.float32)
    motor_target = joint_target.copy()
    motor_torques = joint_torques.copy()

    for transmission in transmissions:
        motor_indices = list(transmission.motor_indices)
        joint_indices = list(transmission.joint_indices)
        motor_pos, motor_torque = transmission.joint_command_to_motor_space(
            joint_target[joint_indices],
            joint_torques[joint_indices],
        )
        motor_target[motor_indices] = motor_pos
        motor_torques[motor_indices] = motor_torque

    return motor_target, motor_torques
