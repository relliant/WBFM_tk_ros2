from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from bodyctrl_msgs.msg import CmdMotorCtrl, CmdSetMotorPosition, MotorCtrl, SetMotorPosition

from .ankle_transmission import (
    LinearJointTransmission,
    apply_joint_command_transmissions,
    apply_motor_state_transmissions,
)
from .fsm import JoystickCommand, decode_joy_message
from .robot_contract import RobotContract, get_robot_contract


@dataclass
class BodyState:
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    dof_torque: np.ndarray
    rpy: np.ndarray
    ang_vel: np.ndarray
    timestamp_sec: float


class JointMap:
    def __init__(
        self,
        can_id_by_index: List[int],
        policy_name_by_index: List[str],
        leg_indices: List[int],
        arm_indices: List[int],
        waist_ids: List[int],
        ankle_transmissions: Optional[List[LinearJointTransmission]] = None,
    ) -> None:
        self.can_id_by_index = list(can_id_by_index)
        self.policy_name_by_index = list(policy_name_by_index)
        self.leg_indices = list(leg_indices)
        self.arm_indices = list(arm_indices)
        self.waist_ids = list(waist_ids)
        self.ankle_transmissions = list(ankle_transmissions or [])
        self.index_by_can_id = {can_id: idx for idx, can_id in enumerate(self.can_id_by_index)}

    @classmethod
    def from_yaml(cls, path: str) -> "JointMap":
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        data = data.get("joint_map", data)
        return cls(
            can_id_by_index=data["can_id_by_index"],
            policy_name_by_index=data["policy_name_by_index"],
            leg_indices=data["leg_indices"],
            arm_indices=data["arm_indices"],
            waist_ids=data.get("waist_ids", [31]),
            ankle_transmissions=[
                LinearJointTransmission.from_dict(item)
                for item in data.get("ankle_transmissions", [])
            ],
        )


class RobotIO:
    def __init__(self, joint_map: JointMap, contract: RobotContract | None = None) -> None:
        self.contract = contract or get_robot_contract("tienkung")
        self.joint_map = joint_map
        self.reset()

    def reset(self) -> None:
        self.dof_pos = np.zeros(self.contract.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.contract.num_actions, dtype=np.float32)
        self.dof_torque = np.zeros(self.contract.num_actions, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_state_time_sec = 0.0
        self.last_joy: Optional[JoystickCommand] = None

    def _update_motor_status(self, msg: Any, stamp_sec: float) -> None:
        for status in getattr(msg, "status", []):
            index = self.joint_map.index_by_can_id.get(int(status.name))
            if index is None:
                continue
            self.dof_pos[index] = float(status.pos)
            self.dof_vel[index] = float(status.speed)
            self.dof_torque[index] = float(getattr(status, "current", 0.0))
        self.dof_pos, self.dof_vel, self.dof_torque = apply_motor_state_transmissions(
            self.dof_pos,
            self.dof_vel,
            self.dof_torque,
            self.joint_map.ankle_transmissions,
        )
        self.last_state_time_sec = max(self.last_state_time_sec, stamp_sec)

    def ingest_leg_status(self, msg: Any, stamp_sec: float) -> None:
        self._update_motor_status(msg, stamp_sec)

    def ingest_arm_status(self, msg: Any, stamp_sec: float) -> None:
        self._update_motor_status(msg, stamp_sec)

    def ingest_imu(self, msg: Any, stamp_sec: float) -> None:
        euler = getattr(msg, "euler", None)
        angular_velocity = getattr(msg, "angular_velocity", None)
        if euler is not None:
            self.rpy[:] = [float(euler.roll), float(euler.pitch), float(euler.yaw)]
        if angular_velocity is not None:
            self.ang_vel[:] = [float(angular_velocity.x), float(angular_velocity.y), float(angular_velocity.z)]
        self.last_state_time_sec = max(self.last_state_time_sec, stamp_sec)

    def ingest_joy(self, msg: Any) -> JoystickCommand:
        self.last_joy = decode_joy_message(list(getattr(msg, "axes", [])), list(getattr(msg, "buttons", [])))
        return self.last_joy

    def snapshot(self) -> BodyState:
        return BodyState(
            dof_pos=self.dof_pos.copy(),
            dof_vel=self.dof_vel.copy(),
            dof_torque=self.dof_torque.copy(),
            rpy=self.rpy.copy(),
            ang_vel=self.ang_vel.copy(),
            timestamp_sec=self.last_state_time_sec,
        )

    def build_command_dict(self, target_dof_pos: np.ndarray, kp: np.ndarray, kd: np.ndarray, torques: Optional[np.ndarray] = None) -> Dict[str, Any]:
        target_dof_pos = np.asarray(target_dof_pos, dtype=np.float32)
        kp = np.asarray(kp, dtype=np.float32)
        kd = np.asarray(kd, dtype=np.float32)
        torques = np.zeros_like(target_dof_pos) if torques is None else np.asarray(torques, dtype=np.float32)
        motor_target_dof_pos, motor_torques = apply_joint_command_transmissions(
            target_dof_pos,
            torques,
            self.joint_map.ankle_transmissions,
        )
        return {
            "leg": [
                {
                    "name": self.joint_map.can_id_by_index[index],
                    "kp": float(kp[index]),
                    "kd": float(kd[index]),
                    "pos": float(motor_target_dof_pos[index]),
                    "spd": 0.0,
                    "tor": float(motor_torques[index]),
                }
                for index in self.joint_map.leg_indices
            ],
            "arm": [
                {
                    "name": self.joint_map.can_id_by_index[index],
                    "kp": float(kp[index]),
                    "kd": float(kd[index]),
                    "pos": float(motor_target_dof_pos[index]),
                    "spd": 0.0,
                    "tor": float(motor_torques[index]),
                }
                for index in self.joint_map.arm_indices
            ],
            "waist": [
                {
                    "name": waist_id,
                    "pos": 0.0,
                    "spd": 0.3,
                    "cur": 3.0,
                }
                for waist_id in self.joint_map.waist_ids
            ],
        }

    def build_ros_messages(self, node: Any, command_dict: Dict[str, Any]) -> tuple[CmdMotorCtrl, CmdMotorCtrl, CmdSetMotorPosition]:
        leg_msg = CmdMotorCtrl()
        arm_msg = CmdMotorCtrl()
        waist_msg = CmdSetMotorPosition()
        stamp = node.get_clock().now().to_msg()
        leg_msg.header.stamp = stamp
        arm_msg.header.stamp = stamp
        waist_msg.header.stamp = stamp

        for item in command_dict["leg"]:
            cmd = MotorCtrl()
            cmd.name = int(item["name"])
            cmd.kp = float(item["kp"])
            cmd.kd = float(item["kd"])
            cmd.pos = float(item["pos"])
            cmd.spd = float(item["spd"])
            cmd.tor = float(item["tor"])
            leg_msg.cmds.append(cmd)

        for item in command_dict["arm"]:
            cmd = MotorCtrl()
            cmd.name = int(item["name"])
            cmd.kp = float(item["kp"])
            cmd.kd = float(item["kd"])
            cmd.pos = float(item["pos"])
            cmd.spd = float(item["spd"])
            cmd.tor = float(item["tor"])
            arm_msg.cmds.append(cmd)

        for item in command_dict["waist"]:
            cmd = SetMotorPosition()
            cmd.name = int(item["name"])
            cmd.pos = float(item["pos"])
            cmd.spd = float(item["spd"])
            cmd.cur = float(item["cur"])
            waist_msg.cmds.append(cmd)

        return leg_msg, arm_msg, waist_msg


def default_gains(contract: RobotContract | None = None) -> tuple[np.ndarray, np.ndarray]:
    contract = contract or get_robot_contract("tienkung")
    kp = np.array([
        700.0, 700.0, 500.0, 700.0, 15.0, 15.0,
        700.0, 700.0, 500.0, 700.0, 15.0, 15.0,
        60.0, 20.0, 10.0, 10.0,
        60.0, 20.0, 10.0, 10.0,
    ], dtype=np.float32)
    kd = np.array([
        20.0, 20.0, 15.0, 10.0, 1.25, 1.25,
        20.0, 20.0, 15.0, 10.0, 1.25, 1.25,
        3.0, 1.5, 1.0, 1.0,
        3.0, 1.5, 1.0, 1.0,
    ], dtype=np.float32)
    return kp, kd
