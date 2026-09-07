"""Bounded, task-aware prefix-feature replay memory."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile

import numpy as np

REPLAY_KEYS = ("prefix_tokens", "prefix_mask", "state", "actions")


class FeatureReplayBuffer:
    """Per-skill reservoir followed by a bounded global reservoir.

    Each entry contains only the frozen prefix embedding, its validity mask,
    proprioceptive state, and action target. Raw observations and diffusion
    variables are intentionally rejected.
    """

    def __init__(self, capacity: int = 500, per_task_quota: int = 50, seed: int = 0):
        if capacity <= 0 or per_task_quota <= 0:
            raise ValueError("capacity and per_task_quota must be positive.")
        self.capacity = capacity
        self.per_task_quota = per_task_quota
        self._rng = np.random.default_rng(seed)
        self._entries: list[dict[str, np.ndarray]] = []
        self._task_ids: list[int] = []
        self._seen_by_task: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self._task_ids)))

    def add(self, task_id: int, sample: Mapping[str, np.ndarray]) -> bool:
        unknown = set(sample) - set(REPLAY_KEYS)
        missing = set(REPLAY_KEYS) - set(sample)
        if unknown or missing:
            raise ValueError(f"Replay sample keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        entry = {key: np.asarray(sample[key]) for key in REPLAY_KEYS}
        entry["prefix_tokens"] = np.asarray(entry["prefix_tokens"], dtype=np.float16)
        entry["prefix_mask"] = np.asarray(entry["prefix_mask"], dtype=bool)
        entry["state"] = np.asarray(entry["state"], dtype=np.float32)
        entry["actions"] = np.asarray(entry["actions"], dtype=np.float32)
        if entry["prefix_tokens"].ndim != 2 or entry["prefix_mask"].ndim != 1:
            raise ValueError("A replay entry must not include a batch dimension.")

        task_id = int(task_id)
        seen = self._seen_by_task.get(task_id, 0) + 1
        self._seen_by_task[task_id] = seen
        same_task = [index for index, value in enumerate(self._task_ids) if value == task_id]

        if len(same_task) < self.per_task_quota:
            if len(self._entries) < self.capacity:
                self._entries.append(entry)
                self._task_ids.append(task_id)
                return True
            replace = int(self._rng.integers(0, len(self._entries)))
            self._entries[replace], self._task_ids[replace] = entry, task_id
            return True

        chosen = int(self._rng.integers(0, seen))
        if chosen >= self.per_task_quota:
            return False
        replace = same_task[chosen]
        self._entries[replace] = entry
        return True

    def sample(self, batch_size: int, *, exclude_task: int | None = None) -> dict[str, np.ndarray]:
        eligible = [i for i, task_id in enumerate(self._task_ids) if task_id != exclude_task]
        if not eligible:
            raise ValueError("No eligible replay samples are available.")
        indices = self._rng.choice(eligible, size=batch_size, replace=len(eligible) < batch_size)
        return {key: np.stack([self._entries[int(index)][key] for index in indices]) for key in REPLAY_KEYS}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "version": 1,
                "capacity": self.capacity,
                "per_task_quota": self.per_task_quota,
                "task_ids": self._task_ids,
                "seen_by_task": self._seen_by_task,
                "rng_state": self._rng.bit_generator.state,
            }
        )
        arrays = (
            {key: np.stack([entry[key] for entry in self._entries]) for key in REPLAY_KEYS} if self._entries else {}
        )
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez(temporary, metadata=np.asarray(metadata), **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> FeatureReplayBuffer:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("version") != 1:
                raise ValueError(f"Unsupported replay format version: {metadata.get('version')}")
            buffer = cls(metadata["capacity"], metadata["per_task_quota"])
            buffer._task_ids = [int(value) for value in metadata["task_ids"]]
            buffer._seen_by_task = {int(key): int(value) for key, value in metadata["seen_by_task"].items()}
            buffer._rng.bit_generator.state = metadata["rng_state"]
            buffer._entries = [
                {key: np.asarray(archive[key][index]) for key in REPLAY_KEYS} for index in range(len(buffer._task_ids))
            ]
        return buffer
