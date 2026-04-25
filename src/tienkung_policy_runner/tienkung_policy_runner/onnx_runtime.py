from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class OnnxPolicyRunner:
    def __init__(self, policy_path: str, device: str = "cpu") -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for policy inference but is not installed")
        providers = []
        available = ort.get_available_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        outputs = self.session.run([self.output_name], {self.input_name: obs})
        return np.asarray(outputs[0], dtype=np.float32)


def inspect_policy(policy_path: str) -> Optional[dict]:
    if ort is None:
        return None
    session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    return {
        "input_name": session.get_inputs()[0].name,
        "input_shape": list(session.get_inputs()[0].shape),
        "output_name": session.get_outputs()[0].name,
        "output_shape": list(session.get_outputs()[0].shape),
        "providers": session.get_providers(),
    }
