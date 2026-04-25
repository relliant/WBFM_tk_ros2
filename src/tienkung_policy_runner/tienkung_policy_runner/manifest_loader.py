from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .robot_contract import RobotContract, get_robot_contract


REQUIRED_KEYS = {
    "robot",
    "num_actions",
    "n_mimic_obs",
    "n_proprio",
    "n_obs_single",
    "history_len",
    "total_obs_size",
    "policy_frequency_hz",
    "default_dof_pos",
    "action_scale",
    "ankle_indices",
    "joint_order_policy",
}


def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}
    missing = sorted(REQUIRED_KEYS - manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    return manifest


def validate_manifest_against_contract(manifest: Dict[str, Any], contract: RobotContract | None = None) -> RobotContract:
    robot = manifest["robot"]
    contract = contract or get_robot_contract(robot)
    if robot != contract.name:
        raise ValueError(f"Manifest robot {robot!r} does not match contract {contract.name!r}")

    scalar_fields = [
        "num_actions",
        "n_mimic_obs",
        "n_proprio",
        "n_obs_single",
        "history_len",
        "total_obs_size",
        "policy_frequency_hz",
    ]
    for field in scalar_fields:
        if int(manifest[field]) != int(getattr(contract, field)):
            raise ValueError(f"Manifest field {field}={manifest[field]} does not match contract {getattr(contract, field)}")

    expected_lengths = {
        "default_dof_pos": contract.num_actions,
        "action_scale": contract.num_actions,
        "ankle_indices": len(contract.ankle_indices),
        "joint_order_policy": contract.num_actions,
    }
    for field, expected in expected_lengths.items():
        actual = len(manifest[field])
        if actual != expected:
            raise ValueError(f"Manifest field {field} length {actual} does not match expected {expected}")

    return contract
