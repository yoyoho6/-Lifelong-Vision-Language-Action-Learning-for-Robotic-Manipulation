import jax.numpy as jnp
import numpy as np
import pytest

from openpi.lifelong_vla.replay_buffer import FeatureReplayBuffer


def _sample(value: float) -> dict[str, np.ndarray]:
    return {
        "prefix_tokens": np.full((3, 4), value, dtype=np.float16),
        "prefix_mask": np.ones((3,), dtype=bool),
        "state": np.full((2,), value, dtype=np.float32),
        "actions": np.full((5, 2), value, dtype=np.float32),
    }


def test_task_quota_sampling_and_roundtrip(tmp_path):
    buffer = FeatureReplayBuffer(capacity=4, per_task_quota=2, seed=7)
    for value in range(10):
        buffer.add(0, _sample(float(value)))
    buffer.add(1, _sample(100.0))

    assert len(buffer) == 3
    assert buffer.task_ids == (0, 1)
    replay = buffer.sample(3, exclude_task=1)
    assert replay["prefix_tokens"].shape == (3, 3, 4)
    assert np.all(replay["actions"] < 100)

    path = tmp_path / "replay.npz"
    buffer.save(path)
    restored = FeatureReplayBuffer.load(path)
    assert len(restored) == len(buffer)
    assert restored.task_ids == buffer.task_ids
    assert restored.sample(1)["state"].shape == (1, 2)


def test_rejects_diffusion_variables():
    buffer = FeatureReplayBuffer()
    invalid = {**_sample(1.0), "noise": np.zeros((5, 2))}
    with pytest.raises(ValueError, match="unknown"):
        buffer.add(0, invalid)


def test_bfloat_prefix_is_persisted_as_numeric_float16(tmp_path):
    sample = _sample(1.0)
    sample["prefix_tokens"] = np.asarray(jnp.ones((3, 4), dtype=jnp.bfloat16))
    buffer = FeatureReplayBuffer(capacity=1, per_task_quota=1)
    buffer.add(0, sample)
    path = tmp_path / "bfloat.npz"
    buffer.save(path)

    restored = FeatureReplayBuffer.load(path).sample(1)
    assert restored["prefix_tokens"].dtype == np.float16
