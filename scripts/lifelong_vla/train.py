"""Train the isolated LifelongVLA branch.

Run ``uv run scripts/lifelong_vla/train.py --help`` for all options.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
from flax import struct
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

from openpi.lifelong_vla.config import LifelongPi0Config
from openpi.lifelong_vla.model import LifelongPi0
from openpi.lifelong_vla.replay_buffer import FeatureReplayBuffer
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    base_config: str
    exp_name: str
    task_sequence: tuple[int, ...] | None = None
    steps_per_task: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    short_rank: int = 16
    long_rank: int = 16
    lora_alpha: float = 16.0
    gate_bias: float = 0.0
    replay_capacity: int = 500
    cache_per_task: int = 50
    replay_batch_size: int = 4
    replay_weight: float = 1.0
    distill_weight: float = 0.1
    long_lr_scale: float = 0.1
    log_interval: int | None = None
    save_interval: int | None = None
    seed: int | None = None
    fsdp_devices: int | None = None
    wandb_enabled: bool | None = None
    overwrite: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        if self.overwrite and self.resume:
            raise ValueError("--overwrite and --resume cannot be enabled together.")
        if self.replay_weight < 0 or self.distill_weight < 0:
            raise ValueError("Replay and distillation weights must be non-negative.")
        if not 0 < self.long_lr_scale <= 1:
            raise ValueError("long_lr_scale must be in (0, 1].")


@struct.dataclass
class LifelongTrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    teacher_params: nnx.State
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    ema_decay: float | None = struct.field(pytree_node=False, default=None)
    ema_params: nnx.State | None = None


def _build_config(args: TrainArgs) -> tuple[_config.TrainConfig, tuple[int, ...], int]:
    base = _config.get_config(args.base_config)
    if not isinstance(base.model, pi0_config.Pi0Config):
        raise TypeError("LifelongVLA currently supports pi0/Pi0Config base configurations only.")
    sequence = args.task_sequence or base.libero_task_sequence
    if not sequence:
        raise ValueError("Provide --task-sequence or use a base config with libero_task_sequence.")
    sequence = tuple(int(task_id) for task_id in sequence)
    if len(set(sequence)) != len(sequence):
        raise ValueError(f"Duplicate task ids are not supported: {sequence}")
    steps_per_task = args.steps_per_task or base.steps_per_task
    if steps_per_task <= 0:
        raise ValueError("steps_per_task must be positive.")

    model = LifelongPi0Config(
        dtype=base.model.dtype,
        paligemma_variant=str(base.model.paligemma_variant).removesuffix("_lora"),
        action_expert_variant=str(base.model.action_expert_variant).removesuffix("_lora"),
        action_dim=base.model.action_dim,
        action_horizon=base.model.action_horizon,
        max_token_len=base.model.max_token_len,
        pi05=base.model.pi05,
        discrete_state_input=base.model.discrete_state_input,
        short_rank=args.short_rank,
        long_rank=args.long_rank,
        lora_alpha=args.lora_alpha,
        gate_bias=args.gate_bias,
    )
    overrides: dict[str, Any] = {
        "model": model,
        "freeze_filter": model.get_freeze_filter(),
        "ema_decay": None,
        "exp_name": args.exp_name,
        "libero_incremental": True,
        "libero_task_sequence": sequence,
        "steps_per_task": steps_per_task,
        "num_train_steps": len(sequence) * steps_per_task,
        "overwrite": args.overwrite,
        "resume": args.resume,
    }
    for name in ("batch_size", "num_workers", "log_interval", "save_interval", "seed", "fsdp_devices", "wandb_enabled"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return dataclasses.replace(base, **overrides), sequence, steps_per_task


def _load_weights(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def init_train_state(config: _config.TrainConfig, rng: at.KeyArrayLike, mesh, *, resume: bool):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(init_rng, partial_params=None):
        model_rng, _ = jax.random.split(init_rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, model_state = nnx.split(model)
            model_state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, model_state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16))
        )
        trainable = params.filter(config.trainable_filter)
        return LifelongTrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=tx.init(trainable),
            teacher_params=trainable,
            tx=tx,
        )

    shape = jax.eval_shape(init, rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    if resume:
        return shape, state_sharding
    partial = _load_weights(config.weight_loader, shape.params.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(init, donate_argnums=(1,), in_shardings=replicated, out_shardings=state_sharding)(rng, partial)
    return state, state_sharding


def _apply_update(config, state, model, grads, long_lr_scale):
    long_filter = nnx_utils.PathRegex(".*lora_long.*")
    grads = nnx_utils.state_map(
        grads, long_filter, lambda parameter: parameter.replace(parameter.value * long_lr_scale)
    )
    trainable = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, trainable)
    nnx.update(model, optax.apply_updates(trainable, updates))
    return dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state), grads


def current_step(config, long_lr_scale, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions = batch

    def loss_fn(differentiable_model: LifelongPi0):
        losses, features = differentiable_model.current_loss_and_features(rng, observation, actions, train=True)
        return jnp.mean(losses), features

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, features), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    state, grads = _apply_update(config, state, model, grads, long_lr_scale)
    metrics = {
        "loss": loss,
        "loss_new": loss,
        "loss_replay": 0.0,
        "loss_distill": 0.0,
        "grad_norm": optax.global_norm(grads),
    }
    return state, metrics, features


def replay_step(config, args, rng, state, batch, replay):
    model = nnx.merge(state.model_def, state.params)
    teacher = nnx.merge(state.model_def, state.params)
    nnx.update(teacher, state.teacher_params)
    model.train()
    teacher.eval()
    observation, actions = batch
    current_rng, replay_rng = jax.random.split(rng)
    replay_noise, replay_time = model.sample_diffusion(replay_rng, replay["actions"])
    teacher_prediction, _ = teacher.replay_prediction(replay, replay_noise, replay_time, gradient_mode="both")
    teacher_prediction = jax.lax.stop_gradient(teacher_prediction)

    def loss_fn(differentiable_model: LifelongPi0):
        new_losses, features = differentiable_model.current_loss_and_features(
            current_rng, observation, actions, train=True
        )
        replay_prediction, replay_target = differentiable_model.replay_prediction(
            replay, replay_noise, replay_time, gradient_mode="long"
        )
        new_loss = jnp.mean(new_losses)
        replay_loss = jnp.mean(jnp.square(replay_prediction - replay_target))
        distill_loss = jnp.mean(jnp.square(replay_prediction - teacher_prediction))
        total = new_loss + args.replay_weight * replay_loss + args.distill_weight * distill_loss
        return total, (new_loss, replay_loss, distill_loss, features)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, auxiliary), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    new_loss, replay_loss, distill_loss, features = auxiliary
    state, grads = _apply_update(config, state, model, grads, args.long_lr_scale)
    metrics = {
        "loss": loss,
        "loss_new": new_loss,
        "loss_replay": replay_loss,
        "loss_distill": distill_loss,
        "grad_norm": optax.global_norm(grads),
    }
    return state, metrics, features


def _task_config(config: _config.TrainConfig, task_id: int) -> _config.TrainConfig:
    if not hasattr(config.data, "libero_task_indices"):
        raise TypeError(f"Data factory {type(config.data).__name__} does not support task filtering.")
    data = dataclasses.replace(config.data, libero_task_indices=[task_id], libero_replay_episodes=None)
    return dataclasses.replace(config, data=data, libero_task_indices=[task_id], libero_replay_episodes=None)


def _collection_schedule(steps: int, batch_size: int, quota: int, seed: int) -> dict[int, list[int]]:
    count = min(steps * batch_size, quota)
    selected = np.random.default_rng(seed).choice(steps * batch_size, size=count, replace=False)
    schedule: dict[int, list[int]] = {}
    for flat_index in selected:
        step, batch_index = divmod(int(flat_index), batch_size)
        schedule.setdefault(step, []).append(batch_index)
    return schedule


def _save(config, manager, state, loader, replay_buffer, step):
    _checkpoints.save_state(manager, state, loader, step)
    replay_path = config.checkpoint_dir / "lifelong_replay" / f"{step}.npz"
    replay_buffer.save(replay_path)


def _init_wandb(config, resuming):
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
        return
    id_path = config.checkpoint_dir / "wandb_id.txt"
    if resuming and id_path.exists():
        wandb.init(id=id_path.read_text().strip(), resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        id_path.write_text(wandb.run.id)


def run(args: TrainArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Running LifelongVLA on %s", platform.node())
    config, task_sequence, steps_per_task = _build_config(args)
    if config.batch_size % jax.device_count() != 0:
        raise ValueError("Global batch size must be divisible by the JAX device count.")

    cache_dir = os.environ.get("OPENPI_JAX_CACHE_DIR", "/tmp/openpi_lifelong_vla_jax_cache")
    epath.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    _init_wandb(config, resuming)
    state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    first_task_config = _task_config(config, task_sequence[0])
    restore_loader = _data_loader.create_data_loader(first_task_config, sharding=data_sharding, shuffle=True)
    if resuming:
        state = _checkpoints.restore_state(manager, state, restore_loader)
        replay_path = config.checkpoint_dir / "lifelong_replay" / f"{int(state.step)}.npz"
        if not replay_path.exists():
            raise FileNotFoundError(f"Missing replay sidecar for resumed checkpoint: {replay_path}")
        replay_buffer = FeatureReplayBuffer.load(replay_path)
    else:
        replay_buffer = FeatureReplayBuffer(args.replay_capacity, args.cache_per_task, config.seed)

    current_jit = jax.jit(
        functools.partial(current_step, config, args.long_lr_scale),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated, None),
        donate_argnums=(1,),
    )
    replay_jit = jax.jit(
        functools.partial(replay_step, config, args),
        in_shardings=(replicated, state_sharding, data_sharding, None),
        out_shardings=(state_sharding, replicated, None),
        donate_argnums=(1,),
    )

    global_step = int(state.step)
    start_task_position, start_local_step = divmod(global_step, steps_per_task)
    for task_position in range(start_task_position, len(task_sequence)):
        task_id = task_sequence[task_position]
        local_start = start_local_step if task_position == start_task_position else 0
        task_config = _task_config(config, task_id)
        loader = _data_loader.create_data_loader(task_config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        schedule = _collection_schedule(
            steps_per_task, config.batch_size, args.cache_per_task, config.seed + task_position
        )
        progress = tqdm.tqdm(
            range(local_start, steps_per_task), initial=local_start, total=steps_per_task, desc=f"task {task_id}"
        )
        for local_step in progress:
            batch = next(iterator)
            old_available = any(buffer_task != task_id for buffer_task in replay_buffer.task_ids)
            train_rng, step_rng = jax.random.split(train_rng)
            with sharding.set_mesh(mesh):
                if old_available:
                    replay_host = replay_buffer.sample(args.replay_batch_size, exclude_task=task_id)
                    replay_device = jax.tree.map(jnp.asarray, replay_host)
                    state, metrics, features = replay_jit(step_rng, state, batch, replay_device)
                else:
                    state, metrics, features = current_jit(step_rng, state, batch)
            global_step += 1

            for batch_index in schedule.get(local_step, []):
                sample = {key: np.asarray(jax.device_get(value[batch_index])) for key, value in features.items()}
                replay_buffer.add(task_id, sample)

            if global_step % config.log_interval == 0:
                values = {key: float(value) for key, value in jax.device_get(metrics).items()}
                progress.write(
                    f"global_step={global_step} task={task_id} "
                    + " ".join(f"{key}={value:.5f}" for key, value in values.items())
                )
                wandb.log({**values, "task_id": task_id, "replay_size": len(replay_buffer)}, step=global_step)

            at_task_boundary = local_step + 1 == steps_per_task
            if config.save_interval > 0 and global_step % config.save_interval == 0 and not at_task_boundary:
                _save(config, manager, state, loader, replay_buffer, global_step)

        state = dataclasses.replace(state, teacher_params=state.params.filter(config.trainable_filter))
        _save(config, manager, state, loader, replay_buffer, global_step)
        start_local_step = 0

    manager.wait_until_finished()
    logging.info("Training complete at step %d; replay entries=%d", global_step, len(replay_buffer))


if __name__ == "__main__":
    run(tyro.cli(TrainArgs))
