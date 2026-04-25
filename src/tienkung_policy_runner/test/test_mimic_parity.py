import numpy as np

from tienkung_motion_source.motion_lib_adapter import DEFAULT_MIMIC_OBS_TIENKUNG


def test_default_mimic_shape():
    assert DEFAULT_MIMIC_OBS_TIENKUNG.shape == (26,)
    np.testing.assert_allclose(DEFAULT_MIMIC_OBS_TIENKUNG[:6], np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
