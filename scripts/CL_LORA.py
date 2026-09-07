import dataclasses
import functools
import logging
from logging import config  # noqa: F401
import platform
import collections
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
        wandb.init(mode="disabled")
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

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# ======================================
# Debug: 看一个原始 batch，确认 task 过滤是否生效
# ======================================

def debug_one_batch(config: _config.TrainConfig):
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=False,
        framework="pytorch",
    )

    inner_loader = data_loader._data_loader
    if hasattr(inner_loader, "torch_loader"):
        torch_loader = inner_loader.torch_loader
    else:
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
# Host-side ReplayBuffer (cache 存在 host)
# ======================================

class ReplayBuffer:
    """
    存 model.compute_loss_and_cache() 返回的 cache（已 stop_gradient，且通常是 bf16/fp16）。
    做 ring buffer，并支持随机采样拼 batch。
    """
    def __init__(self, maxlen: int):
        self.buf = collections.deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buf)

    def add(self, cache_host: dict[str, np.ndarray], take_n: int = 1):
        small = {}
        for k, v in cache_host.items():
            small[k] = v[:take_n]
        self.buf.append(small)

    def can_sample(self, bs: int) -> bool:
        return (len(self.buf) > 0) and (bs > 0)

    def sample(self, bs: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, len(self.buf), size=bs)
        items = [self.buf[i] for i in idx]
        out = {}
        for k in items[0].keys():
            out[k] = np.concatenate([it[k] for it in items], axis=0)
        return out


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
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
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

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# --------------------------------------
# 原始 step（joint training 仍然用它）
# --------------------------------------
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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


# --------------------------------------
# step_new（当前任务数据）= 更新 + 返回 cache
# --------------------------------------
@at.typecheck
def train_step_new(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array], dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions = batch
    train_rng = jax.random.fold_in(rng, state.step)

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss, cache = model.compute_loss_and_cache(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), cache
    
    #@at.typecheck
    #def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        #chunked_loss, cache = model.compute_loss_and_cache(rng, observation, actions, train=True)

        # --- 关键：检查 chunked_loss 是否已经非有限 ---
        #is_finite = jnp.isfinite(chunked_loss)
        #jax.debug.print(
            #"chunked_loss finite? {} | min={} max={} mean={}",
            #jnp.all(is_finite),
            #jnp.nanmin(chunked_loss),
            #jnp.nanmax(chunked_loss),
            #jnp.nanmean(chunked_loss),
        #)

    # 如果你怀疑是 denom=0（mask=0）导致的 NaN，通常表现为：
    # finite? False 且 min/max/mean 里出现 nan/inf

        #return jnp.mean(chunked_loss), cache

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, cache), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss_new": loss,
        "grad_norm_new": optax.global_norm(grads),
        "param_norm_new": optax.global_norm(kernel_params),
    }
    return new_state, info, cache
    #return state, info, cache


# --------------------------------------
# step_old（旧 cache 重放）= 继续更新同一个 state（方案A）
# --------------------------------------
@at.typecheck
def train_step_old(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    replay_cache: dict[str, at.Array],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)

    @at.typecheck
    def loss_fn(model: _model.BaseModel, replay_cache: dict[str, at.Array]):
        chunked_loss = model.compute_loss_from_cache(replay_cache)
        return jnp.mean(chunked_loss)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, replay_cache)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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

    info = {
        "loss_old": loss,
        "grad_norm_old": optax.global_norm(grads),
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
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ======================================
# LIBERO 增量训练（方案A：new 连续 10 步，再做 1 步 old）
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
    # ---- 永远拿到非空 task_sequence（不再 assert None）----
    task_sequence = getattr(config, "libero_task_sequence", None)

    if task_sequence is None:
        data_idx = getattr(getattr(config, "data", None), "libero_task_indices", None)
        if data_idx is not None and len(data_idx) > 0:
            task_sequence = tuple(data_idx)
            logging.warning(
                "config.libero_task_sequence is None; fallback to config.data.libero_task_indices=%s",
                task_sequence,
            )

    if task_sequence is None:
        task_sequence = (34, 36)
        logging.warning(
            "config.libero_task_sequence is None (and no fallback found). "
            "Force default task_sequence=%s to avoid None error.",
            task_sequence,
        )

    if isinstance(task_sequence, list):
        task_sequence = tuple(task_sequence)

    if not isinstance(task_sequence, tuple) or len(task_sequence) == 0:
        raise ValueError(f"Invalid task_sequence={task_sequence!r}. Expected non-empty tuple/list of ints.")

    logging.info("Final task_sequence=%s", task_sequence)

    # ---------- replay 超参 ----------
    REPLAY_MAXLEN = 450
    REPLAY_BS = 4
    CACHE_TAKE_N = 1
    CACHE_STORE_INTERVAL = 200

    # new 10 步 -> old 1 步
    DO_REPLAY_EVERY = 200
    # 可选：等 buffer 至少有这么多条再启用 replay
    MIN_REPLAY_FILL = 50

    replay = ReplayBuffer(maxlen=REPLAY_MAXLEN)

    # 方案A：单 state（交替更新同一 state）
    state = train_state

    # jit: step_new donate state only
    pstep_new = jax.jit(
        functools.partial(train_step_new, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding, None),
        donate_argnums=(1,),
    )

    # jit: step_old donate state only
    pstep_old = jax.jit(
        functools.partial(train_step_old, config),
        in_shardings=(replicated_sharding, train_state_sharding, None),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_step = int(state.step) if not isinstance(state.step, jax.Array) else int(jax.device_get(state.step))

    for task_id in task_sequence:
        logging.info("=== Training LIBERO task %d ===", task_id)

        # 你之前要求的 debug 输出（保留）
        print("=== 检查属性层级 ===")
        print(f"0. config.libero_task_sequence (getattr): {getattr(config, 'libero_task_sequence', None)}")
        print(f"1. config 是否有 libero_task_indices: {hasattr(config, 'libero_task_indices')}")
        print(f"2. config.data 是否有 libero_task_indices: {hasattr(config.data, 'libero_task_indices')}")

        task_data_config = dataclasses.replace(config.data, libero_task_indices=[task_id])
        task_config = dataclasses.replace(config, data=task_data_config)

        data_loader = _data_loader.create_data_loader(
            task_config,
            sharding=data_sharding,
            shuffle=True,
        )
        data_iter = iter(data_loader)

        for local_step in range(config.steps_per_task):
            batch = next(data_iter)

            # 每次迭代 split 3 份 rng：更新 train_rng；new/old 各一份
            with sharding.set_mesh(mesh):
                train_rng, step_rng_new, step_rng_old = jax.random.split(train_rng, 3)

            # --------- (A) new step：每步都做 ---------
            with sharding.set_mesh(mesh):
                state, m_new, cache_cur = pstep_new(step_rng_new, state, batch)

            # 存 cache（非常稀疏：10000 步一次）
            if (global_step % CACHE_STORE_INTERVAL) == 0:
                cache_host = jax.device_get(cache_cur)
                replay.add(cache_host, take_n=CACHE_TAKE_N)

            # --------- (B) old step：每 10 步做一次 ---------
            do_old = (global_step % DO_REPLAY_EVERY == 0) and (len(replay) >= MIN_REPLAY_FILL)
            if do_old and replay.can_sample(REPLAY_BS):
                replay_cache_host = replay.sample(REPLAY_BS)
                replay_cache_dev = jax.tree.map(lambda x: jnp.asarray(x), replay_cache_host)
                with sharding.set_mesh(mesh):
                    state, m_old = pstep_old(step_rng_old, state, replay_cache_dev)
            else:
                m_old = {
                    "loss_old": jnp.array(0.0, dtype=jnp.float32),
                    "grad_norm_old": jnp.array(0.0, dtype=jnp.float32),
                }

            global_step += 1

            # --------- logging ---------
            if global_step % config.log_interval == 0:
                log_new = jax.device_get(m_new)
                log_old = jax.device_get(m_old)
                logging.info(
                    "[task %d] global_step=%d local_step=%d "
                    "loss_new=%.6f loss_old=%.6f grad_new=%.6f grad_old=%.6f (do_old=%s buf=%d)",
                    task_id,
                    global_step,
                    local_step,
                    float(log_new["loss_new"]),
                    float(log_old["loss_old"]),
                    float(log_new["grad_norm_new"]),
                    float(log_old["grad_norm_old"]),
                    str(bool(do_old)),
                    int(len(replay)),
                )
                wandb.log(
                    {
                        "loss_new": float(log_new["loss_new"]),
                        "loss_old": float(log_old["loss_old"]),
                        "grad_norm_new": float(log_new["grad_norm_new"]),
                        "grad_norm_old": float(log_old["grad_norm_old"]),
                        "task_id": int(task_id),
                        "replay_buf_len": int(len(replay)),
                        "do_old": int(bool(do_old)),
                    },
                    step=global_step,
                )

            # 保存：保存当前 state（单 state）
            if (config.save_interval > 0) and (global_step % config.save_interval == 0):
                _checkpoints.save_state(checkpoint_manager, state, data_loader, global_step)

        # task 边界保存
        _checkpoints.save_state(checkpoint_manager, state, data_loader, global_step)

    logging.info("Incremental training finished; waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ======================================
# main
# ======================================

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    jax.config.update("jax_debug_infs", True)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    cache_dir = "/media/ubuntu/data/home/hy/code/openpi-main/data/jax_cache"
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

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_state_or_shape, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if config.libero_incremental:
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
            resuming=False,
        )


if __name__ == "__main__":
    main(_config.cli())
    # 需要单独检查某个 task 的数据时，可以手动调用：
    # cfg = _config.cli()
    # cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, libero_task_indices=[0]))
    # debug_one_batch(cfg)
