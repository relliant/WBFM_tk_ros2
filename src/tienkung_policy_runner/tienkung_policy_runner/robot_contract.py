from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class RobotContract:
    name: str
    num_actions: int
    n_mimic_obs: int
    n_proprio: int
    n_obs_single: int
    history_len: int
    total_obs_size: int
    default_dof_pos: np.ndarray
    action_scale: np.ndarray
    ankle_indices: List[int]
    policy_frequency_hz: int
    joint_order_policy: List[str]


TIENKUNG_JOINT_ORDER = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

ROBOT_CONTRACTS: Dict[str, RobotContract] = {
    "tienkung": RobotContract(
        name="tienkung",
        num_actions=20,
        n_mimic_obs=26,
        n_proprio=65,
        n_obs_single=91,
        history_len=10,
        total_obs_size=1027,
        default_dof_pos=np.array([
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
            0.0, 0.1, 0.0, -0.3,
            0.0, -0.1, 0.0, -0.3,
        ], dtype=np.float32),
        action_scale=np.full(20, 0.5, dtype=np.float32),
        ankle_indices=[4, 5, 10, 11],
        policy_frequency_hz=50,
        joint_order_policy=TIENKUNG_JOINT_ORDER,
    )
}


def get_robot_contract(name: str) -> RobotContract:
    try:
        return ROBOT_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown robot contract: {name}") from exc


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]
