import dataclasses
import functools
import logging
from logging import config
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental  # noqa: F401
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# ======================================
# Logging & wandb
# ======================================

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# ======================================
# Debug: 看一个原始 batch，确认 task 过滤是否生效
# ======================================

def debug_one_batch(config: _config.TrainConfig):
    """
    辅助调试：直接从 DataLoader 里取一个 raw batch，检查 task_index / episode_index 等字段，
    用来确认 task 过滤是不是生效了。
    """
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=False,
        framework="pytorch",  # 很重要：不走 JAX sharding，拿到纯 torch batch
    )

    inner_loader = data_loader._data_loader
    if hasattr(inner_loader, "torch_loader"):
        torch_loader = inner_loader.torch_loader  # TorchDataLoader
    else:
        # RLDSDataLoader：本身就可以直接 __iter__
        torch_loader = inner_loader

    batch = next(iter(torch_loader))

    try:
        keys = batch.keys()
    except AttributeError:
        print("Batch type:", type(batch))
        return

    print("Batch keys:", keys)

    for k in ("task_index", "episode_index", "frame_index"):
        if k in batch:
            v = batch[k]
            try:
                import torch

                print(f"{k} (first 16):", v[:16])
                print(f"unique {k}:", torch.unique(v))
            except Exception:
                print(f"{k} (first 16):", v[:16])

    print("Shapes:")
    for k, v in batch.items():
        try:
            print(" ", k, v.shape)
        except AttributeError:
            print(" ", k, type(v))


# ======================================
# Train state init & step
# ======================================

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState | Any, Any]:
    """Initialize TrainState or just its shape (if resume=True)."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16 for frozen params.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 先只算 shape，方便做 FSDP sharding 以及 checkpoint 恢复
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 恢复模式下，外面会用这个 shape 去 restore checkpoint
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step; pjit 封装在外面."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params_full = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params_full, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params_full
            ),
        )

    # Filter out params that aren't kernels for param_norm.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# ======================================
# Joint training（原版）
# ======================================

def run_joint_training(
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    train_state: training_utils.TrainState,
    train_state_sharding: jax.sharding.Sharding,
    data_loader: _data_loader.DataLoader,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    checkpoint_manager,
    train_rng: at.KeyArrayLike,
    *,
    resuming: bool,
):
    """原始 joint 训练（所有任务混在一起）."""
    data_iter = iter(data_loader)
    batch = next(data_iter)

    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))
    logging.info("First batch obs:\n%s", training_utils.array_tree_to_info(batch[0]))

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos: list[dict[str, at.Array]] = []

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_rng, step_rng = jax.random.split(train_rng)
            train_state, info = ptrain_step(step_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if config.wandb_enabled:
                wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ======================================
# LIBERO 增量训练
# ======================================

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
    LIBERO 增量学习：顺序训练多个 task，共享同一个 TrainState。

    语义：
    - 当前任务：全量 episode
    - 所有旧任务：每个任务随机采样 4 个 episode 做 replay
    - config.libero_task_sequence: 顺序任务列表
    - config.steps_per_task: 每个任务训练步数
    - config.data.libero_task_indices: 当前 dataloader 要加载的 task 列表
    - config.data.libero_replay_episodes: 旧任务 -> 允许的 episode_index 列表
    """
    assert config.libero_task_sequence is not None, "libero_task_sequence must not be None"

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_step = int(train_state.step)

    # 用固定随机种子控制 replay episode 采样
    replay_rng = np.random.default_rng(config.seed)

    # 先把“每个 task 对应哪些 episode_index”统计出来，只做一次
    resolved_data_config = config.data.create(config.assets_dirs, config.model)
    episodes_by_task = _data_loader.get_libero_episode_ids_by_task(
        resolved_data_config,
        action_horizon=config.model.action_horizon,
    )

    logging.info("LIBERO available episode ids by task: %s", episodes_by_task)
    

    for task_pos, task_id in enumerate(config.libero_task_sequence):
        task_id = int(task_id)
        old_task_ids = [int(t) for t in config.libero_task_sequence[:task_pos]]

        logging.info("=== Training LIBERO task %d ===", task_id)
        logging.info("Old tasks before current task %d: %s", task_id, old_task_ids)

        # 为每个旧任务随机采样 4 个 episode
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

        # 当前任务 + 所有旧任务
        # 当前任务不在 replay_episode_map 中 => 保留全部 episode
        task_ids_for_loader = [task_id] + old_task_ids
        task_ids_for_loader = list(dict.fromkeys(task_ids_for_loader))  # 去重保序

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

                if config.wandb_enabled:
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


# ======================================
# main
# ======================================

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    #jax.config.update("jax_debug_nans", True) # Debug 模式下打开 NaN 检测
    jax.config.update("jax_debug_infs", False)
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    cache_dir = os.environ.get("OPENPI_JAX_CACHE_DIR", "/tmp/jax_cache")
    epath.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Checkpoint 管理
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 初始化（或者只构建 shape，用于 resume）
    train_state_or_shape, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if config.libero_incremental:
        # 简化处理：目前不支持从 checkpoint 恢复增量训练，直接重新初始化
        if resuming:
            logging.warning(
                "Resuming with libero_incremental=True is not fully supported yet; starting from scratch instead."
            )
            train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=False)
        else:
            train_state = train_state_or_shape  # type: ignore[assignment]

        run_incremental_training(
            config=config,
            mesh=mesh,
            train_state=train_state,
            train_state_sharding=train_state_sharding,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            checkpoint_manager=checkpoint_manager,
            train_rng=train_rng,
        )
    else:
        # 原始 joint 训练路径
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
        )

        if resuming:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state_or_shape, data_loader)
        else:
            train_state = train_state_or_shape  # type: ignore[assignment]

        run_joint_training(
            config=config,
            mesh=mesh,
            train_state=train_state,
            train_state_sharding=train_state_sharding,
            data_loader=data_loader,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            checkpoint_manager=checkpoint_manager,
            train_rng=train_rng,
            resuming=False,  # 上面已经 restore 过
        )


if __name__ == "__main__":
    main(_config.cli())
    # 需要单独检查某个 task 的数据时，可以手动调用：
    # cfg = _config.cli()
    # cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, libero_task_indices=[0]))
    # debug_one_batch(cfg)
