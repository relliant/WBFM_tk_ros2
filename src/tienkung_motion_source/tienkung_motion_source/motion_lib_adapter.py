from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from tienkung_policy_runner.robot_contract import get_robot_contract
except ImportError:
    from tienkung_policy_runner.tienkung_policy_runner.robot_contract import get_robot_contract


DEFAULT_MIMIC_OBS_TIENKUNG = np.concatenate([
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([1.0], dtype=np.float32),
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
    np.array([
        0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
        0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
        0.0, 0.1, 0.0, -0.3,
        0.0, -0.1, 0.0, -0.3,
    ], dtype=np.float32),
]).astype(np.float32)


def quat_to_euler_wxyz(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = quat / np.linalg.norm(quat)
    vq = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float32)
    return quat_multiply_wxyz(quat_multiply_wxyz(quat_conjugate_wxyz(q), vq), q)[1:]


@dataclass
class MotionFrame:
    root_pos: np.ndarray
    root_rot: np.ndarray
    root_vel: np.ndarray
    root_ang_vel: np.ndarray
    dof_pos: np.ndarray


class MotionLibAdapter:
    def __init__(self, motion_file: str | Path) -> None:
        self.path = Path(motion_file)
        with self.path.open("rb") as file:
            motion_data = pickle.load(file)
        self.fps = float(motion_data["fps"])
        self.root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
        self.root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
        self.dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
        self.num_frames = self.root_pos.shape[0]
        self.dt = 1.0 / self.fps
        self.length_sec = self.dt * max(0, self.num_frames - 1)
        self.root_vel = np.gradient(self.root_pos, self.dt, axis=0).astype(np.float32)
        self.root_ang_vel = self._estimate_root_ang_vel()
        self.contract = get_robot_contract("tienkung")

    def _estimate_root_ang_vel(self) -> np.ndarray:
        eulers = np.stack([quat_to_euler_wxyz(q) for q in self.root_rot], axis=0)
        return np.gradient(eulers, self.dt, axis=0).astype(np.float32)

    def get_motion_length(self) -> float:
        return self.length_sec

    def _blend_indices(self, t_sec: float) -> tuple[int, int, float]:
        if self.length_sec <= 0.0:
            return 0, 0, 0.0
        t = t_sec % self.length_sec
        phase = np.clip(t / self.length_sec, 0.0, 1.0)
        idx0 = int(phase * (self.num_frames - 1))
        idx1 = min(idx0 + 1, self.num_frames - 1)
        blend = phase * (self.num_frames - 1) - idx0
        return idx0, idx1, float(blend)

    def sample_frame(self, t_sec: float) -> MotionFrame:
        idx0, idx1, blend = self._blend_indices(t_sec)
        root_pos = (1.0 - blend) * self.root_pos[idx0] + blend * self.root_pos[idx1]
        root_rot = (1.0 - blend) * self.root_rot[idx0] + blend * self.root_rot[idx1]
        root_rot = root_rot / np.linalg.norm(root_rot)
        dof_pos = (1.0 - blend) * self.dof_pos[idx0] + blend * self.dof_pos[idx1]
        root_vel = self.root_vel[idx0]
        root_ang_vel = self.root_ang_vel[idx0]
        return MotionFrame(root_pos=root_pos, root_rot=root_rot, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_pos=dof_pos)


def build_mimic_obs(adapter: MotionLibAdapter, t_step: int, control_dt: float, future_steps: Sequence[int]) -> np.ndarray:
    future_steps = list(future_steps)
    rows = []
    for step in future_steps:
        t_sec = (t_step + int(step)) * control_dt
        frame = adapter.sample_frame(t_sec)
        euler = quat_to_euler_wxyz(frame.root_rot)
        root_vel_local = quat_rotate_inverse_wxyz(frame.root_rot, frame.root_vel)
        root_ang_vel_local = quat_rotate_inverse_wxyz(frame.root_rot, frame.root_ang_vel)
        row = np.concatenate([
            root_vel_local[:2],
            frame.root_pos[2:3],
            euler[:2],
            root_ang_vel_local[2:3],
            frame.dof_pos,
        ]).astype(np.float32)
        rows.append(row)
    return np.concatenate(rows, axis=0).astype(np.float32)
