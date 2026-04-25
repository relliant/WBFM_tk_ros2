import numpy as np

from tienkung_policy_runner.ankle_transmission import (
    LinearJointTransmission,
    apply_joint_command_transmissions,
    apply_motor_state_transmissions,
)


def test_ankle_zero_target_maps_to_equal_motor_positions():
    transmission = LinearJointTransmission(
        name="left_ankle",
        motor_indices=(4, 5),
        joint_indices=(4, 5),
        motor_to_joint=np.array([
            [0.5, 0.5],
            [0.5, -0.5],
        ], dtype=np.float32),
    )

    joint_target = np.zeros(20, dtype=np.float32)
    joint_torque = np.zeros(20, dtype=np.float32)
    joint_target[4] = -0.5
    joint_target[5] = 0.0

    motor_target, motor_torque = apply_joint_command_transmissions(
        joint_target,
        joint_torque,
        [transmission],
    )

    np.testing.assert_allclose(motor_target[4:6], np.array([-0.5, -0.5], dtype=np.float32))
    np.testing.assert_allclose(motor_torque[4:6], np.zeros(2, dtype=np.float32))


def test_motor_state_round_trip_matches_joint_space_expectation():
    transmission = LinearJointTransmission(
        name="left_ankle",
        motor_indices=(4, 5),
        joint_indices=(4, 5),
        motor_to_joint=np.array([
            [0.5, 0.5],
            [0.5, -0.5],
        ], dtype=np.float32),
    )

    motor_pos = np.zeros(20, dtype=np.float32)
    motor_vel = np.zeros(20, dtype=np.float32)
    motor_torque = np.zeros(20, dtype=np.float32)
    motor_pos[4:6] = [-0.6, -0.4]
    motor_vel[4:6] = [1.0, -0.5]
    motor_torque[4:6] = [2.0, 0.5]

    joint_pos, joint_vel, joint_torque = apply_motor_state_transmissions(
        motor_pos,
        motor_vel,
        motor_torque,
        [transmission],
    )

    np.testing.assert_allclose(joint_pos[4:6], np.array([-0.5, -0.1], dtype=np.float32))
    np.testing.assert_allclose(joint_vel[4:6], np.array([0.25, 0.75], dtype=np.float32))

    round_trip_motor_pos, round_trip_motor_torque = apply_joint_command_transmissions(
        joint_pos,
        joint_torque,
        [transmission],
    )

    np.testing.assert_allclose(round_trip_motor_pos[4:6], motor_pos[4:6])
    np.testing.assert_allclose(round_trip_motor_torque[4:6], motor_torque[4:6])
