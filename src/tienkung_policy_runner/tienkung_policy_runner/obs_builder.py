from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .robot_contract import RobotContract, get_robot_contract


@dataclass
class TienkungObservationBuilder:
    contract: RobotContract = field(default_factory=lambda: get_robot_contract("tienkung"))
    history: Deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.contract.history_len)
        self.reset()

    def reset(self) -> None:
        self.history.clear()
        zeros = np.zeros(self.contract.n_obs_single, dtype=np.float32)
        for _ in range(self.contract.history_len):
            self.history.append(zeros.copy())

    def build_proprio(
        self,
        ang_vel: np.ndarray,
        rpy: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        ang_vel = np.asarray(ang_vel, dtype=np.float32)
        rpy = np.asarray(rpy, dtype=np.float32)
        dof_pos = np.asarray(dof_pos, dtype=np.float32)
        dof_vel = np.asarray(dof_vel, dtype=np.float32)
        last_action = np.asarray(last_action, dtype=np.float32)

        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[self.contract.ankle_indices] = 0.0
        proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            dof_pos - self.contract.default_dof_pos,
            obs_body_dof_vel * 0.05,
            last_action,
        ]).astype(np.float32)
        if proprio.shape[0] != self.contract.n_proprio:
            raise ValueError(f"Expected proprio dim {self.contract.n_proprio}, got {proprio.shape[0]}")
        return proprio

    def build_observation(
        self,
        action_mimic: np.ndarray,
        ang_vel: np.ndarray,
        rpy: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        action_mimic = np.asarray(action_mimic, dtype=np.float32)
        if action_mimic.shape[0] != self.contract.n_mimic_obs:
            raise ValueError(f"Expected mimic dim {self.contract.n_mimic_obs}, got {action_mimic.shape[0]}")
        proprio = self.build_proprio(ang_vel, rpy, dof_pos, dof_vel, last_action)
        current_obs = np.concatenate([action_mimic, proprio]).astype(np.float32)
        if current_obs.shape[0] != self.contract.n_obs_single:
            raise ValueError(f"Expected obs_single dim {self.contract.n_obs_single}, got {current_obs.shape[0]}")
        history_flat = np.asarray(self.history, dtype=np.float32).reshape(-1)
        self.history.append(current_obs.copy())
        final_obs = np.concatenate([current_obs, history_flat, action_mimic]).astype(np.float32)
        if final_obs.shape[0] != self.contract.total_obs_size:
            raise ValueError(f"Expected total obs dim {self.contract.total_obs_size}, got {final_obs.shape[0]}")
        return final_obs


def postprocess_action(raw_action: np.ndarray, contract: RobotContract | None = None) -> np.ndarray:
    contract = contract or get_robot_contract("tienkung")
    raw_action = np.asarray(raw_action, dtype=np.float32)
    clipped = np.clip(raw_action, -10.0, 10.0)
    scaled = clipped * contract.action_scale
    return (scaled + contract.default_dof_pos).astype(np.float32)
