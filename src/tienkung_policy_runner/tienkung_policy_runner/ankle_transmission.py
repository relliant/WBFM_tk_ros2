from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np


class JointTransmission(Protocol):
    name: str
    motor_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]

    def motor_state_to_joint_state(
        self,
        motor_pos: Sequence[float],
        motor_vel: Sequence[float],
        motor_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def joint_command_to_motor_space(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        joint_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _resolve_repo_native_dir() -> Path:
    return Path(__file__).resolve().parent / "native"


def _as_square_matrix(values: Sequence[Sequence[float]], size: int, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (size, size):
        raise ValueError(f"Expected {name} shape {(size, size)}, got {matrix.shape}")
    return matrix


@dataclass
class LinearJointTransmission:
    name: str
    motor_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]
    motor_to_joint: np.ndarray

    def __post_init__(self) -> None:
        self.motor_indices = tuple(int(index) for index in self.motor_indices)
        self.joint_indices = tuple(int(index) for index in self.joint_indices)
        if len(self.motor_indices) != len(self.joint_indices):
            raise ValueError("motor_indices and joint_indices must have the same length")
        size = len(self.motor_indices)
        self.motor_to_joint = _as_square_matrix(self.motor_to_joint, size, "motor_to_joint")
        self.joint_to_motor = np.linalg.inv(self.motor_to_joint).astype(np.float32)
        self.motor_torque_to_joint = np.linalg.inv(self.motor_to_joint).T.astype(np.float32)
        self.joint_torque_to_motor = self.motor_to_joint.T.astype(np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "LinearJointTransmission":
        motor_indices = tuple(data["motor_indices"])
        return cls(
            name=str(data["name"]),
            motor_indices=motor_indices,
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
        joint_vel: Sequence[float],
        joint_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        joint_torque = np.asarray(joint_torque, dtype=np.float32)
        return (
            (self.joint_to_motor @ joint_pos).astype(np.float32),
            (self.joint_to_motor @ joint_vel).astype(np.float32),
            (self.joint_torque_to_motor @ joint_torque).astype(np.float32),
        )


class _FuncSPTransNativeLib:
    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._library.funcsptrans_create.restype = ctypes.c_void_p
        self._library.funcsptrans_destroy.argtypes = [ctypes.c_void_p]
        self._library.funcsptrans_state_to_joint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.funcsptrans_state_to_joint.restype = ctypes.c_int
        self._library.funcsptrans_joint_to_motor.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.funcsptrans_joint_to_motor.restype = ctypes.c_int

    def create_handle(self) -> ctypes.c_void_p:
        handle = self._library.funcsptrans_create()
        if not handle:
            raise RuntimeError("funcSPTrans create failed")
        return ctypes.c_void_p(handle)

    def destroy_handle(self, handle: ctypes.c_void_p) -> None:
        if handle:
            self._library.funcsptrans_destroy(handle)

    def state_to_joint(
        self,
        handle: ctypes.c_void_p,
        motor_pos: np.ndarray,
        motor_vel: np.ndarray,
        motor_torque: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_s = np.zeros(4, dtype=np.float64)
        qd_s = np.zeros(4, dtype=np.float64)
        tau_s = np.zeros(4, dtype=np.float64)
        status = self._library.funcsptrans_state_to_joint(
            handle,
            motor_pos.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            motor_vel.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            motor_torque.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            q_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qd_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tau_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"funcSPTrans state_to_joint failed with code {status}")
        return q_s.astype(np.float32), qd_s.astype(np.float32), tau_s.astype(np.float32)

    def joint_to_motor(
        self,
        handle: ctypes.c_void_p,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_torque: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_p = np.zeros(4, dtype=np.float64)
        qd_p = np.zeros(4, dtype=np.float64)
        tau_p = np.zeros(4, dtype=np.float64)
        status = self._library.funcsptrans_joint_to_motor(
            handle,
            joint_pos.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            joint_vel.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            joint_torque.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            q_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qd_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tau_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"funcSPTrans joint_to_motor failed with code {status}")
        return q_p.astype(np.float32), qd_p.astype(np.float32), tau_p.astype(np.float32)


def _default_funcsptrans_library_candidates() -> list[Path]:
    native_dir = _resolve_repo_native_dir()
    return [
        Path(os.environ["TIENKUNG_FUNCSPTRANS_LIB"])
        if "TIENKUNG_FUNCSPTRANS_LIB" in os.environ
        else None,
        native_dir / "lib/libfuncSPTrans.so",
        native_dir / "lib/libfuncSPTrans.so.1",
        native_dir / "lib/libfuncSPTrans.so.1.0.0",
        Path.home() / "Downloads/ros_lite/src/rl_control_new/src/plugins/p2s/lib/libfuncSPTrans.so",
        Path("/opt/ros_lite/lib/libfuncSPTrans.so"),
    ]


def _detect_eigen_include(explicit: str | None) -> Path:
    candidates = [explicit] if explicit else []
    candidates.extend([
        os.environ.get("TIENKUNG_EIGEN3_INCLUDE"),
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
    ])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "Eigen/Dense").exists():
            return path
    raise FileNotFoundError("Could not find Eigen headers for funcSPTrans wrapper build")


def _build_wrapper_if_needed(wrapper_path: Path, funcsptrans_library: Path, eigen_include: str | None) -> Path:
    if wrapper_path.exists():
        return wrapper_path
    if platform.system() != "Linux":
        raise RuntimeError("funcSPTrans native wrapper can only be built on Linux")

    native_dir = _resolve_repo_native_dir()
    source_path = native_dir / "funcsptrans_c_api.cpp"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing wrapper source: {source_path}")
    if not funcsptrans_library.exists():
        raise FileNotFoundError(f"Missing funcSPTrans library: {funcsptrans_library}")

    compiler = os.environ.get("CXX", "g++")
    eigen_path = _detect_eigen_include(eigen_include)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(source_path),
        "-o",
        str(wrapper_path),
        "-I",
        str(native_dir / "third_party"),
        "-I",
        str(eigen_path),
        "-L",
        str(funcsptrans_library.parent),
        "-Wl,-rpath," + str(funcsptrans_library.parent),
        "-lfuncSPTrans",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return wrapper_path


def _load_native_library(
    wrapper_library: str | None,
    funcsptrans_library: str | None,
    eigen_include: str | None,
) -> _FuncSPTransNativeLib:
    wrapper_path = Path(wrapper_library) if wrapper_library else (_resolve_repo_native_dir() / "build/libfuncsptrans_c_api.so")

    if wrapper_path.exists():
        return _FuncSPTransNativeLib(ctypes.CDLL(str(wrapper_path)))

    candidates = [Path(funcsptrans_library)] if funcsptrans_library else []
    candidates.extend([path for path in _default_funcsptrans_library_candidates() if path is not None])
    for candidate in candidates:
        if candidate.exists():
            built = _build_wrapper_if_needed(wrapper_path, candidate, eigen_include)
            return _FuncSPTransNativeLib(ctypes.CDLL(str(built)))
    raise FileNotFoundError("Could not locate libfuncSPTrans.so for native ankle conversion")


@dataclass
class NativeFuncSPTransTransmission:
    name: str
    motor_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]
    wrapper_library: str = ""
    funcsptrans_library: str = ""
    eigen_include: str = ""
    require_native: bool = True

    def __post_init__(self) -> None:
        self.motor_indices = tuple(int(index) for index in self.motor_indices)
        self.joint_indices = tuple(int(index) for index in self.joint_indices)
        self._native_lib = _load_native_library(
            self.wrapper_library or None,
            self.funcsptrans_library or None,
            self.eigen_include or None,
        )
        self._native_handle = self._native_lib.create_handle()

    @classmethod
    def from_dict(cls, data: dict) -> "NativeFuncSPTransTransmission":
        return cls(
            name=str(data["name"]),
            motor_indices=tuple(data["motor_indices"]),
            joint_indices=tuple(data["joint_indices"]),
            wrapper_library=str(data.get("wrapper_library", "")),
            funcsptrans_library=str(data.get("funcsptrans_library", "")),
            eigen_include=str(data.get("eigen_include", "")),
            require_native=bool(data.get("require_native", True)),
        )

    def __del__(self) -> None:
        if self._native_lib is not None and self._native_handle is not None:
            self._native_lib.destroy_handle(self._native_handle)
            self._native_handle = None

    def motor_state_to_joint_state(
        self,
        motor_pos: Sequence[float],
        motor_vel: Sequence[float],
        motor_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._native_lib.state_to_joint(
            self._native_handle,
            np.asarray(motor_pos, dtype=np.float32),
            np.asarray(motor_vel, dtype=np.float32),
            np.asarray(motor_torque, dtype=np.float32),
        )

    def joint_command_to_motor_space(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        joint_torque: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._native_lib.joint_to_motor(
            self._native_handle,
            np.asarray(joint_pos, dtype=np.float32),
            np.asarray(joint_vel, dtype=np.float32),
            np.asarray(joint_torque, dtype=np.float32),
        )


def transmission_from_dict(data: dict) -> JointTransmission:
    kind = str(data.get("kind", "linear"))
    if kind == "linear":
        return LinearJointTransmission.from_dict(data)
    if kind == "native_funcsptrans":
        return NativeFuncSPTransTransmission.from_dict(data)
    raise ValueError(f"Unsupported joint transmission kind: {kind}")


def apply_motor_state_transmissions(
    dof_pos: Sequence[float],
    dof_vel: Sequence[float],
    dof_torque: Sequence[float],
    transmissions: Iterable[JointTransmission],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos = np.asarray(dof_pos, dtype=np.float32).copy()
    joint_vel = np.asarray(dof_vel, dtype=np.float32).copy()
    joint_torque = np.asarray(dof_torque, dtype=np.float32).copy()
    raw_pos = joint_pos.copy()
    raw_vel = joint_vel.copy()
    raw_torque = joint_torque.copy()

    for transmission in transmissions:
        motor_indices = list(transmission.motor_indices)
        joint_indices = list(transmission.joint_indices)
        pos_block, vel_block, torque_block = transmission.motor_state_to_joint_state(
            raw_pos[motor_indices],
            raw_vel[motor_indices],
            raw_torque[motor_indices],
        )
        joint_pos[joint_indices] = pos_block
        joint_vel[joint_indices] = vel_block
        joint_torque[joint_indices] = torque_block

    return joint_pos, joint_vel, joint_torque


def apply_joint_command_transmissions(
    target_dof_pos: Sequence[float],
    target_dof_vel: Sequence[float],
    torques: Sequence[float],
    transmissions: Iterable[JointTransmission],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_target = np.asarray(target_dof_pos, dtype=np.float32)
    joint_vel = np.asarray(target_dof_vel, dtype=np.float32)
    joint_torques = np.asarray(torques, dtype=np.float32)
    motor_target = joint_target.copy()
    motor_vel = joint_vel.copy()
    motor_torques = joint_torques.copy()

    for transmission in transmissions:
        motor_indices = list(transmission.motor_indices)
        joint_indices = list(transmission.joint_indices)
        pos_block, vel_block, torque_block = transmission.joint_command_to_motor_space(
            joint_target[joint_indices],
            joint_vel[joint_indices],
            joint_torques[joint_indices],
        )
        motor_target[motor_indices] = pos_block
        motor_vel[motor_indices] = vel_block
        motor_torques[motor_indices] = torque_block

    return motor_target, motor_vel, motor_torques
