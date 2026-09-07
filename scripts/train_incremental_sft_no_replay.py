"""Sequential SFT for xarm_data_heyao_2 without experience replay.

Each task gets its own data loader and that loader contains only the current
task.  The model and optimizer state are carried from one task to the next,
which makes this incremental SFT rather than joint training.

Example:
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train_incremental_sft_no_replay.py \
        pi0_xarm_data_heyao_2_incremental_lora \
        --exp-name=xarm_data_heyao_2_incremental_sft_no_replay \
        --batch-size=1 --num-workers=2 --prefetch-factor=2
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import train as _train


EXPECTED_CONFIG_NAME = "pi0_xarm_data_heyao_2_incremental_lora"
EXPECTED_REPO_ID = "xarm_data_heyao_2"


def _init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _validate_config(config: _config.TrainConfig) -> tuple[int, ...]:
    """Fail early if this script is accidentally used with a different dataset."""
    if config.name != EXPECTED_CONFIG_NAME:
        raise ValueError(
            f"This entry point only supports config {EXPECTED_CONFIG_NAME!r}; got {config.name!r}."
        )
    if not config.libero_incremental:
        raise ValueError("libero_incremental must be true for sequential SFT.")
    if not config.libero_task_sequence:
        raise ValueError("libero_task_sequence must contain at least one task id.")
    if config.steps_per_task <= 0:
        raise ValueError(f"steps_per_task must be positive; got {config.steps_per_task}.")

    task_sequence = tuple(int(task_id) for task_id in config.libero_task_sequence)
    if len(set(task_sequence)) != len(task_sequence):
        raise ValueError(f"libero_task_sequence contains duplicate task ids: {task_sequence}.")

    resolved_data_config = config.data.create(config.assets_dirs, config.model)
    if resolved_data_config.repo_id != EXPECTED_REPO_ID:
        raise ValueError(
            f"This entry point requires repo_id={EXPECTED_REPO_ID!r}; "
            f"got {resolved_data_config.repo_id!r}."
        )

    expected_steps = len(task_sequence) * config.steps_per_task
    if config.num_train_steps != expected_steps:
        logging.warning(
            "num_train_steps=%d is ignored by sequential SFT; running %d tasks x %d steps = %d steps",
            config.num_train_steps,
            len(task_sequence),
            config.steps_per_task,
            expected_steps,
        )
    return task_sequence


def _config_for_current_task(config: _config.TrainConfig, task_id: int) -> _config.TrainConfig:
    """Return a config whose dataset can expose only one task and no old episodes."""
    task_data_config = dataclasses.replace(
        config.data,
        libero_task_indices=[int(task_id)],
        libero_replay_episodes=None,
    )
    return dataclasses.replace(
        config,
        data=task_data_config,
        libero_task_indices=[int(task_id)],
        libero_replay_episodes=None,
    )


def _create_current_task_loader(
    config: _config.TrainConfig,
    task_id: int,
    data_sharding: jax.sharding.Sharding,
) -> _data_loader.DataLoader:
    task_config = _config_for_current_task(config, task_id)
    data_loader = _data_loader.create_data_loader(
        task_config,
        sharding=data_sharding,
        shuffle=True,
    )

    # Keep this assertion beside loader creation so future data-pipeline changes
    # cannot silently turn this entry point into joint/replay training.
    resolved = data_loader.data_config()
    selected_tasks = getattr(resolved, "libero_task_indices", None)
    old_episode_selection = getattr(resolved, "libero_replay_episodes", None)
    if selected_tasks != [int(task_id)] or old_episode_selection not in (None, {}):
        raise RuntimeError(
            "No-replay invariant violated: "
            f"selected_tasks={selected_tasks!r}, old_episode_selection={old_episode_selection!r}."
        )
    return data_loader


def run_incremental_sft_no_replay(
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    train_state: training_utils.TrainState,
    train_state_sharding: jax.sharding.Sharding,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    checkpoint_manager: Any,
    train_rng: at.KeyArrayLike,
    task_sequence: tuple[int, ...],
) -> training_utils.TrainState:
    """Train tasks in sequence while retaining no samples from earlier tasks."""
    ptrain_step = jax.jit(
        functools.partial(_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_step = int(train_state.step)
    total_steps = len(task_sequence) * config.steps_per_task
    if global_step > total_steps:
        raise ValueError(
            f"Checkpoint step {global_step} exceeds configured no-replay training length {total_steps}."
        )

    for task_position, task_id in enumerate(task_sequence):
        task_start_step = task_position * config.steps_per_task
        task_end_step = task_start_step + config.steps_per_task
        if global_step >= task_end_step:
            logging.info("Skipping completed task %d (ends at global step %d)", task_id, task_end_step)
            continue

        local_start_step = max(0, global_step - task_start_step)
        logging.info(
            "=== Sequential SFT task %d (%d/%d), local step %d/%d; current-task data only ===",
            task_id,
            task_position + 1,
            len(task_sequence),
            local_start_step,
            config.steps_per_task,
        )
        data_loader = _create_current_task_loader(config, task_id, data_sharding)
        data_iter = iter(data_loader)
        infos: list[dict[str, at.Array]] = []

        pbar = tqdm.tqdm(
            range(local_start_step, config.steps_per_task),
            initial=local_start_step,
            total=config.steps_per_task,
            dynamic_ncols=True,
            desc=f"task {task_id}",
        )
        for local_step in pbar:
            batch = next(data_iter)
            with sharding.set_mesh(mesh):
                train_rng, step_rng = jax.random.split(train_rng)
                train_state, metrics = ptrain_step(step_rng, train_state, batch)

            global_step += 1
            infos.append(metrics)

            if global_step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items())
                pbar.write(
                    f"task={task_id} global_step={global_step} local_step={local_step + 1}: {info_str}"
                )
                if config.wandb_enabled:
                    wandb.log(
                        {
                            **reduced_info,
                            "task_id": int(task_id),
                            "task_position": task_position,
                        },
                        step=global_step,
                    )
                infos = []

            should_save_interval = config.save_interval > 0 and global_step % config.save_interval == 0
            should_save_task_boundary = global_step == task_end_step
            if should_save_interval or should_save_task_boundary:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, global_step)

    logging.info("Sequential no-replay SFT finished at global step %d", global_step)
    checkpoint_manager.wait_until_finished()
    return train_state


def main(config: _config.TrainConfig) -> None:
    _init_logging()
    logging.info("Running on: %s", platform.node())
    jax.config.update("jax_debug_infs", False)
    task_sequence = _validate_config(config)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}."
        )

    cache_dir = os.environ.get("OPENPI_JAX_CACHE_DIR", "/tmp/jax_cache")
    epath.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    _train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_state, train_state_sharding = _train.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        restore_loader = _create_current_task_loader(config, task_sequence[0], data_sharding)
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, restore_loader)
        logging.info("Restored checkpoint at global step %d", int(train_state.step))
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    run_incremental_sft_no_replay(
        config=config,
        mesh=mesh,
        train_state=train_state,
        train_state_sharding=train_state_sharding,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        checkpoint_manager=checkpoint_manager,
        train_rng=train_rng,
        task_sequence=task_sequence,
    )


if __name__ == "__main__":
    main(_config.cli())
