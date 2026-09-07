# openpi/training/data_loader.py
from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import inspect
import logging
import multiprocessing
import os
import typing
from pathlib import Path
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


# =========================
# Protocols
# =========================
class Dataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co: ...
    def __len__(self) -> int: ...


class IterableDataset(Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


class DataLoader(Protocol[T_co]):
    def data_config(self) -> _config.DataConfig: ...
    def __iter__(self) -> Iterator[T_co]: ...


# =========================
# Dataset wrappers
# =========================
class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                batch_size = next(v.shape[0] for v in sample.values())
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023
                transformed = [self._transform(s) for s in individual_samples]
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


# =========================
# Fake dataset
# =========================
class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]  # remove batch dim
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        return {**observation.to_dict(), "actions": action}

    def __len__(self) -> int:
        return self._num_samples


# =========================
# Pickle-safe helpers
# =========================
def _as_py_int(x: Any) -> int | None:
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return int(x.view(-1)[0].item())
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            return int(x.reshape(-1)[0])
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None
            return _as_py_int(x[0])
        return int(x)
    except Exception:
        return None


def _as_str(x: Any) -> str | None:
    try:
        if x is None:
            return None
        if isinstance(x, bytes):
            return x.decode("utf-8")
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return str(x.view(-1)[0].item())
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None
            return _as_str(x[0])
        return str(x)
    except Exception:
        return None


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Keep other code behavior unchanged across lerobot versions:
    only pass kwargs that the callable accepts.
    """
    try:
        sig = inspect.signature(fn)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def _get_required_local_root(data_config: Any) -> Path:
    """
    Local-only mode:
    require data_config.root to be provided (no HF cache / snapshot behavior).
    """
    #root = getattr(data_config, "root", None)
    root ="/media/ubuntu/data/home/hy/code/openpi-main/hf_data"
    if root is None or str(root).strip() == "":
        raise ValueError(
            "Local dataset root is required. "
            "Please set data.root to your local LeRobot dataset directory (the folder containing data/ and meta/)."
        )
    return Path(root).expanduser()


def _build_tasks_by_index(dataset_meta: Any) -> tuple[tuple[str, ...], dict[str, int]]:
    """
    Build (tasks_by_index, task_text_to_id) robustly.
    Prefer dataset_meta.task_text_to_id if present, else fallback to dataset_meta.tasks order.
    """
    tasks = list(getattr(dataset_meta, "tasks", []) or [])
    t2i = getattr(dataset_meta, "task_text_to_id", None)

    if isinstance(t2i, dict) and len(t2i) > 0:
        max_id = int(max(t2i.values()))
        tasks_by_index: list[str] = [""] * (max_id + 1)
        for txt, idx in t2i.items():
            if idx is None:
                continue
            i = int(idx)
            if 0 <= i < len(tasks_by_index):
                tasks_by_index[i] = str(txt)
        for i, t in enumerate(tasks):
            if i < len(tasks_by_index) and not tasks_by_index[i]:
                tasks_by_index[i] = str(t)
        tasks_by_index = [t if t else f"<task_{i}>" for i, t in enumerate(tasks_by_index)]
        return tuple(tasks_by_index), {str(k): int(v) for k, v in t2i.items()}

    tasks_by_index = tuple(str(t) for t in tasks)
    t2i_fallback = {str(t): i for i, t in enumerate(tasks_by_index)}
    return tasks_by_index, t2i_fallback


@dataclasses.dataclass(frozen=True)
class CoerceTaskFields:
    """
    Pickle-safe transform:
    - Normalize task_index to Python int
    - Force task text to match task_index (prevents prompt mismatch)
    """
    tasks_by_index: tuple[str, ...]
    task_text_to_id: dict[str, int]

    def __call__(self, sample: dict) -> dict:
        ti = _as_py_int(sample.get("task_index", None))

        # If task_index missing, infer from task/task(s) text
        if ti is None:
            txt = _as_str(sample.get("task", None))
            if txt is None:
                txt = _as_str(sample.get("tasks", None))
            if txt is not None and txt in self.task_text_to_id:
                ti = int(self.task_text_to_id[txt])

        existing_task = _as_str(sample.get("task", None))

        # If we have a valid ti, force alignment (but don't overwrite real text)
        if ti is not None and 0 <= ti < len(self.tasks_by_index):
            sample["task_index"] = int(ti)

            # 关键：已有文本就保留；只有缺失/纯数字才覆盖
            if existing_task is None or existing_task.strip().isdigit():
                sample["task"] = self.tasks_by_index[int(ti)]
        else:
            if ti is not None:
                sample["task_index"] = int(ti)
            if "task" not in sample and "tasks" in sample:
                txt = _as_str(sample.get("tasks", None))
                if txt is not None:
                    sample["task"] = txt


        return sample

@dataclasses.dataclass(frozen=True)
class FixPromptFromTaskText:
    """If prompt is numeric but task has natural-language text, use task as prompt."""
    def __call__(self, sample: dict) -> dict:
        prompt = _as_str(sample.get("prompt", None))
        task_txt = _as_str(sample.get("task", None))

        # Only fix when prompt looks like an index (e.g., "34") but task is real text.
        if prompt is not None and prompt.strip().isdigit():
            if task_txt is not None and (not task_txt.strip().isdigit()) and task_txt.strip() != "":
                sample["prompt"] = task_txt
        return sample



# =========================
# LIBERO task filtering (NO cache)
# =========================
class LiberoTaskFilteredDataset(Dataset):
    """
    Filter by a single task_index WITHOUT persistent cache.
    Prefer HF column access (task_index) to avoid full scan.
    """

    def __init__(self, base_dataset: Dataset, task_index: Any):
        self._dataset = base_dataset

        # ---- MOD 1: robustly coerce task_index (supports int / [int] / tensor / ndarray) ----
        ti = _as_py_int(task_index)
        if ti is None:
            raise ValueError(f"Invalid task_index={task_index!r} (type={type(task_index).__name__})")
        self._task_index = int(ti)
        # -----------------------------------------------------------------------------------

        indices_by_task = self._build_indices_by_task(base_dataset)

        arr = indices_by_task.get(int(self._task_index), None)
        if arr is None or len(arr) == 0:
            logging.warning(
                "[LiberoTaskFilteredDataset] No samples for task %s; available=%s",
                int(self._task_index),
                sorted(indices_by_task.keys()),
            )
            raise ValueError(
                f"No samples found for requested task_index={int(self._task_index)}. "
                f"Available tasks in base_dataset: {sorted(indices_by_task.keys())}"
            )

        self._indices = arr.astype(np.int64)
        logging.info(
            "[LiberoTaskFilteredDataset] Using %d samples for task %d",
            len(self._indices),
            int(self._task_index),
        )

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, index: SupportsIndex):
        real_idx = int(self._indices[int(index)])
        return self._dataset[real_idx]

    @staticmethod
    def _unwrap_dataset(obj: Any) -> Any:
        seen: set[int] = set()
        cur = obj
        while hasattr(cur, "_dataset") and id(cur) not in seen:
            seen.add(id(cur))
            cur = getattr(cur, "_dataset")
        return cur

    @classmethod
    def _try_get_hf_dataset(cls, obj: Any) -> Any | None:
        base = cls._unwrap_dataset(obj)
        for attr in ("hf_dataset", "_hf_dataset"):
            ds = getattr(base, attr, None)
            if ds is not None:
                return ds
        if hasattr(base, "column_names"):
            return base
        return None

    @classmethod
    def _build_indices_by_task(cls, dataset: Dataset) -> dict[int, np.ndarray]:
        hf = cls._try_get_hf_dataset(dataset)
        if hf is not None and hasattr(hf, "column_names"):
            cols = set(hf.column_names)
            if "task_index" in cols:
                col = hf["task_index"]
                col_np = np.asarray([_as_py_int(x) for x in np.asarray(col)], dtype=np.int32)
                uniq = np.unique(col_np)
                out: dict[int, np.ndarray] = {}
                for t in uniq.tolist():
                    out[int(t)] = np.nonzero(col_np == t)[0].astype(np.int64)
                logging.info(
                    "[LiberoTaskFilteredDataset] Built indices_by_task from HF column (tasks=%s)",
                    sorted(out.keys()),
                )
                return out

        # Fallback: one-time scan
        n = len(dataset)
        indices_by_task: dict[int, list[int]] = {}
        for i in range(n):
            s = dataset[i]
            if not isinstance(s, dict):
                continue
            ti = _as_py_int(s.get("task_index", None))
            if ti is None:
                continue
            indices_by_task.setdefault(int(ti), []).append(i)

        if not indices_by_task:
            raise ValueError("No samples contain a valid 'task_index' field; cannot filter.")

        return {t: np.asarray(v, dtype=np.int64) for t, v in indices_by_task.items()}


# =========================
# Dataset creation
# =========================
def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Local-only: require root
    root = _get_required_local_root(data_config)

    # Metadata (local)
    meta_kwargs: dict[str, Any] = dict(root=root)
    meta_kwargs = _filter_kwargs_for_callable(lerobot_dataset.LeRobotDatasetMetadata, meta_kwargs)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, **meta_kwargs)

    tasks_by_index, task_text_to_id = _build_tasks_by_index(dataset_meta)
    tasks_map: dict[int, str] = {i: tasks_by_index[i] for i in range(len(tasks_by_index))}

    # Base dataset (local)
    ds_kwargs: dict[str, Any] = dict(root=root)
    ds_kwargs = _filter_kwargs_for_callable(lerobot_dataset.LeRobotDataset, ds_kwargs)

    base_dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        **ds_kwargs,
    )

    # Coerce fields FIRST (pickle-safe)
    base_dataset = TransformedDataset(
        base_dataset,
        [CoerceTaskFields(tasks_by_index=tasks_by_index, task_text_to_id=task_text_to_id)],
    )

    # Filter by task_index (NO cache)  [single-task semantics: int | None]
    # ---- MOD 2: robustly coerce raw libero_task_indices to a single int (supports list/tensor/etc) ----
    task_index_raw = getattr(data_config, "libero_task_indices", None)
    task_index = _as_py_int(task_index_raw)

    if task_index is not None:
        logging.info(
            "[create_torch_dataset] Filtering LIBERO task_index=%d (raw=%r, type=%s)",
            int(task_index),
            task_index_raw,
            type(task_index_raw).__name__,
        )
        dataset = LiberoTaskFilteredDataset(base_dataset, int(task_index))
    else:
        dataset = base_dataset
        logging.info(
            "[create_torch_dataset] No libero_task_indices; using full dataset. (raw=%r, type=%s)",
            task_index_raw,
            type(task_index_raw).__name__,
        )
    # ------------------------------------------------------------------------------------------------

    # Prompt from task
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [
                _transforms.PromptFromLeRobotTask(tasks_map),
                FixPromptFromTaskText(),
            ],
        )

    if os.environ.get("OPENPI_DEBUG_FIRST_SAMPLE", "0") == "1":
        s0 = dataset[0]
        logging.info(
            "[DEBUG first sample] keys=%s task_index=%r task=%r prompt=%r",
            list(s0.keys()),
            s0.get("task_index", None),
            s0.get("task", None),
            s0.get("prompt", None),
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    kwargs: dict[str, Any] = dict(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )
    if hasattr(data_config, "datasets"):
        kwargs["datasets"] = data_config.datasets
    if hasattr(data_config, "filter_dict_path"):
        kwargs["filter_dict_path"] = data_config.filter_dict_path

    kwargs = _filter_kwargs_for_callable(DroidRldsDataset, kwargs)
    return typing.cast(Dataset, DroidRldsDataset(**kwargs))


# =========================
# Transforms
# =========================
def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    norm_stats: dict[str, Any] = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    norm_stats: dict[str, Any] = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


# =========================
# Data loader creation
# =========================
def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                typing.cast(torch.utils.data.Dataset, dataset),
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info("local_batch_size: %d", local_batch_size)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")

    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)
    return DataLoaderImpl(data_config, data_loader)


# =========================
# TorchDataLoader / RLDSDataLoader
# =========================
class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        seed: int = 0,
        framework: str = "jax",
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than dataset size ({len(dataset)}).")

        self._sharding = sharding
        if sharding is None and framework == "jax":
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1

                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
