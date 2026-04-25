import numpy as np

from tienkung_policy_runner.obs_builder import TienkungObservationBuilder, postprocess_action
from tienkung_policy_runner.robot_contract import get_robot_contract


def test_obs_dimensions_and_zeroed_ankles():
    contract = get_robot_contract('tienkung')
    builder = TienkungObservationBuilder(contract)
    action_mimic = np.zeros(contract.n_mimic_obs, dtype=np.float32)
    ang_vel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rpy = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    dof_pos = contract.default_dof_pos.copy()
    dof_vel = np.arange(contract.num_actions, dtype=np.float32)
    last_action = np.ones(contract.num_actions, dtype=np.float32)

    proprio = builder.build_proprio(ang_vel, rpy, dof_pos, dof_vel, last_action)
    assert proprio.shape == (contract.n_proprio,)
    assert proprio[25 + 4] == 0.0
    assert proprio[25 + 5] == 0.0
    assert proprio[25 + 10] == 0.0
    assert proprio[25 + 11] == 0.0

    obs = builder.build_observation(action_mimic, ang_vel, rpy, dof_pos, dof_vel, last_action)
    assert obs.shape == (contract.total_obs_size,)


def test_action_postprocess_matches_contract():
    contract = get_robot_contract('tienkung')
    raw_action = np.array([20.0] * contract.num_actions, dtype=np.float32)
    target = postprocess_action(raw_action, contract)
    expected = contract.default_dof_pos + contract.action_scale * 10.0
    np.testing.assert_allclose(target, expected)
