import dataclasses
import functools
import logging

import jax
import numpy as np
import wandb

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import final as _final


FINAL2_REPLAY_TASKS = (30, 31, 32, 33, 34, 35, 36, 37, 38)
FINAL2_TRAIN_TASK = 39
FINAL2_EFFECTIVE_SEQUENCE = FINAL2_REPLAY_TASKS + (FINAL2_TRAIN_TASK,)


def run_incremental_training(
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    train_state: training_utils.TrainState,
    train_state_sharding: jax.sharding.Sharding,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    checkpoint_manager,
    train_rng: at.KeyArrayLike,
):
    """
    LIBERO 增量训练的 final2 版本：
    - 固定只训练 task 39
    - task 30~38 只做 replay
    - 每个 replay 任务随机采样 3 个 episode
    """
    ptrain_step = jax.jit(
        functools.partial(_final.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_step = int(train_state.step)
    replay_rng = np.random.default_rng(config.seed)

    resolved_data_config = config.data.create(config.assets_dirs, config.model)
    episodes_by_task = _data_loader.get_libero_episode_ids_by_task(
        resolved_data_config,
        action_horizon=config.model.action_horizon,
    )

    logging.info("LIBERO available episode ids by task: %s", episodes_by_task)
    logging.info(
        "final2 overrides config.libero_task_sequence=%s with fixed sequence=%s",
        getattr(config, "libero_task_sequence", None),
        FINAL2_EFFECTIVE_SEQUENCE,
    )

    task_id = FINAL2_TRAIN_TASK
    old_task_ids = [int(t) for t in FINAL2_REPLAY_TASKS]

    logging.info("=== Training only final LIBERO task %d ===", task_id)
    logging.info("Replay-only tasks before current task %d: %s", task_id, old_task_ids)

    replay_episode_map: dict[int, list[int]] = {}
    for old_task_id in old_task_ids:
        available_eps = episodes_by_task.get(int(old_task_id), [])
        if not available_eps:
            logging.warning(
                "[run_incremental_training] No available episodes found for old task %d; skip replay for it.",
                old_task_id,
            )
            continue

        k = min(3, len(available_eps))
        sampled_eps = replay_rng.choice(
            np.asarray(available_eps, dtype=np.int64),
            size=k,
            replace=False,
        ).tolist()

        replay_episode_map[int(old_task_id)] = [int(x) for x in sampled_eps]

    task_ids_for_loader = [task_id] + old_task_ids

    task_data_config = dataclasses.replace(
        config.data,
        libero_task_indices=task_ids_for_loader,
        libero_replay_episodes=replay_episode_map,
    )
    task_config = dataclasses.replace(config, data=task_data_config)

    logging.info(
        "[run_incremental_training] task=%d loader_tasks=%s replay_episode_map=%s",
        task_id,
        task_ids_for_loader,
        replay_episode_map,
    )

    data_loader = _data_loader.create_data_loader(
        task_config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)

    for local_step in range(config.steps_per_task):
        batch = next(data_iter)

        with sharding.set_mesh(mesh):
            train_rng, step_rng = jax.random.split(train_rng)
            train_state, metrics = ptrain_step(step_rng, train_state, batch)

        global_step += 1

        if global_step % config.log_interval == 0:
            log_metrics = jax.device_get(metrics)

            logging.info(
                "[task %d] global_step=%d local_step=%d "
                "loss=%.6f grad_norm=%.6f param_norm=%.6f",
                task_id,
                global_step,
                local_step,
                float(log_metrics["loss"]),
                float(log_metrics["grad_norm"]),
                float(log_metrics["param_norm"]),
            )

            wandb.log(
                {
                    "loss": float(log_metrics["loss"]),
                    "grad_norm": float(log_metrics["grad_norm"]),
                    "param_norm": float(log_metrics["param_norm"]),
                    "task_id": int(task_id),
                },
                step=global_step,
            )

        if (config.save_interval > 0) and (global_step % config.save_interval == 0):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, global_step)

    logging.info("Incremental training finished; waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


_final.run_incremental_training = run_incremental_training


if __name__ == "__main__":
    _final.main(_config.cli())
