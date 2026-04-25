import numpy as np

from tienkung_policy_runner.ankle_transmission import (
    LinearJointTransmission,
    apply_joint_command_transmissions,
    apply_motor_state_transmissions,
)
from tienkung_policy_runner.motor_calibration import MotorCalibration


def test_block_ankle_zero_target_maps_to_equal_motor_positions_per_side():
    transmission = LinearJointTransmission(
        name="ankle_block",
        motor_indices=(4, 5, 10, 11),
        joint_indices=(4, 5, 10, 11),
        motor_to_joint=np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, -0.5],
        ], dtype=np.float32),
    )

    joint_target = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)
    joint_torque = np.zeros(20, dtype=np.float32)
    joint_target[4] = -0.5
    joint_target[10] = -0.5

    motor_target, motor_vel, motor_torque = apply_joint_command_transmissions(
        joint_target,
        joint_vel,
        joint_torque,
        [transmission],
    )

    np.testing.assert_allclose(motor_target[4:6], np.array([-0.5, -0.5], dtype=np.float32))
    np.testing.assert_allclose(motor_target[10:12], np.array([-0.5, -0.5], dtype=np.float32))
    np.testing.assert_allclose(motor_vel[4:6], np.zeros(2, dtype=np.float32))
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

    round_trip_motor_pos, round_trip_motor_vel, round_trip_motor_torque = apply_joint_command_transmissions(
        joint_pos,
        joint_vel,
        joint_torque,
        [transmission],
    )

    np.testing.assert_allclose(round_trip_motor_pos[4:6], motor_pos[4:6])
    np.testing.assert_allclose(round_trip_motor_vel[4:6], motor_vel[4:6])
    np.testing.assert_allclose(round_trip_motor_torque[4:6], motor_torque[4:6])


def test_motor_calibration_matches_ros_lite_equations():
    calibration = MotorCalibration.from_dict(
        {
            "zero_pos_offset": [0.1, 0.2],
            "zero_offset": [0.0, 0.0],
            "motor_dir": [1.0, -1.0],
            "ct_scale": [2.0, 3.0],
            "wrap_threshold_rad": float(np.pi),
        },
        2,
    )

    q, qdot, tau, zero_cnt = calibration.convert_feedback(
        raw_pos=np.array([0.4, 0.8], dtype=np.float32),
        raw_vel=np.array([1.5, -2.0], dtype=np.float32),
        raw_current=np.array([0.5, 1.0], dtype=np.float32),
        zero_cnt=np.zeros(2, dtype=np.float32),
    )

    np.testing.assert_allclose(q, np.array([0.3, -0.6], dtype=np.float32))
    np.testing.assert_allclose(qdot, np.array([1.5, 2.0], dtype=np.float32))
    np.testing.assert_allclose(tau, np.array([1.0, -3.0], dtype=np.float32))
    np.testing.assert_allclose(zero_cnt, np.zeros(2, dtype=np.float32))

    motor_q, motor_qdot, motor_tau = calibration.convert_command(
        joint_pos=q,
        joint_vel=qdot,
        joint_torque=tau,
        zero_cnt=zero_cnt,
    )

    np.testing.assert_allclose(motor_q, np.array([0.4, 0.8], dtype=np.float32))
    np.testing.assert_allclose(motor_qdot, np.array([1.5, -2.0], dtype=np.float32))
    np.testing.assert_allclose(motor_tau, np.array([1.0, 3.0], dtype=np.float32))
