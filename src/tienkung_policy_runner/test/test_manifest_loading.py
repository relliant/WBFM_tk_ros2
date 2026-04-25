import tempfile
from pathlib import Path

import yaml

from tienkung_policy_runner.manifest_loader import load_manifest, validate_manifest_against_contract
from tienkung_policy_runner.robot_contract import get_robot_contract


def test_manifest_validation_accepts_matching_contract():
    contract = get_robot_contract('tienkung')
    manifest = {
        'robot': contract.name,
        'num_actions': contract.num_actions,
        'n_mimic_obs': contract.n_mimic_obs,
        'n_proprio': contract.n_proprio,
        'n_obs_single': contract.n_obs_single,
        'history_len': contract.history_len,
        'total_obs_size': contract.total_obs_size,
        'policy_frequency_hz': contract.policy_frequency_hz,
        'default_dof_pos': contract.default_dof_pos.tolist(),
        'action_scale': contract.action_scale.tolist(),
        'ankle_indices': contract.ankle_indices,
        'joint_order_policy': contract.joint_order_policy,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'policy_manifest.yaml'
        path.write_text(yaml.safe_dump(manifest), encoding='utf-8')
        loaded = load_manifest(path)
        validated = validate_manifest_against_contract(loaded)
        assert validated.name == 'tienkung'
