"""Offline teacher-forced proxy evaluation for the real xArm dataset.

This cannot measure real-robot task success because no xArm simulator or task
success detector exists in this repository. Instead, it samples observations
from recorded demonstration episodes, predicts an action chunk, and compares
that chunk with the recorded action chunk. Three explicit thresholds turn the
action agreement metrics into strict/nominal/loose proxy success rates.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: str
    config_name: str = "pi0_xarm_data_heyao_2_incremental_lora"
    episodes_per_task: int = 50
    anchors_per_episode: int = 3
    batch_size: int = 5
    diffusion_steps: int = 10
    seed: int = 7
    output_dir: str = "eval_results/xarm_offline_ckpt50000"


THRESHOLDS = {
    "strict": {"position_rmse_mm": 20.0, "rotation_rmse_rad": 0.07, "gripper_accuracy": 0.90},
    "nominal": {"position_rmse_mm": 30.0, "rotation_rmse_rad": 0.10, "gripper_accuracy": 0.80},
    "loose": {"position_rmse_mm": 50.0, "rotation_rmse_rad": 0.15, "gripper_accuracy": 0.70},
}


def _stack_samples(samples: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)


def _select_anchor_indices(
    task_col: np.ndarray,
    episode_col: np.ndarray,
    *,
    episodes_per_task: int,
    anchors_per_episode: int,
) -> tuple[list[int], list[tuple[int, int]], dict[int, list[int]]]:
    selected_indices: list[int] = []
    owners: list[tuple[int, int]] = []
    episodes_by_task: dict[int, list[int]] = {}

    quantiles = np.linspace(0.25, 0.75, anchors_per_episode)
    for task_id in sorted(np.unique(task_col).tolist()):
        task_episodes = sorted(np.unique(episode_col[task_col == task_id]).tolist())[:episodes_per_task]
        episodes_by_task[int(task_id)] = [int(x) for x in task_episodes]
        for episode_id in task_episodes:
            frame_indices = np.flatnonzero((task_col == task_id) & (episode_col == episode_id))
            if len(frame_indices) == 0:
                continue
            anchor_positions = np.rint(quantiles * (len(frame_indices) - 1)).astype(np.int64)
            for anchor_position in anchor_positions:
                selected_indices.append(int(frame_indices[int(anchor_position)]))
                owners.append((int(task_id), int(episode_id)))

    return selected_indices, owners, episodes_by_task


def _passes_threshold(result: dict[str, float], threshold: dict[str, float]) -> bool:
    return (
        result["position_rmse_mm"] <= threshold["position_rmse_mm"]
        and result["rotation_rmse_rad"] <= threshold["rotation_rmse_rad"]
        and result["gripper_accuracy"] >= threshold["gripper_accuracy"]
    )


def evaluate(args: Args) -> dict:
    if args.episodes_per_task <= 0 or args.anchors_per_episode <= 0 or args.batch_size <= 0:
        raise ValueError("episodes_per_task, anchors_per_episode, and batch_size must be positive")

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {checkpoint_dir / 'params'}")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != "xarm_data_heyao_2":
        raise ValueError(f"Expected xarm_data_heyao_2, got repo_id={data_config.repo_id!r}")
    if data_config.asset_id is None:
        raise ValueError("asset_id is required to load checkpoint normalization statistics")

    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(
        data_config,
        norm_stats=norm_stats,
        libero_task_indices=None,
        libero_replay_episodes=None,
    )

    logging.info("Building xArm dataset index")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    _, task_col, episode_col = _data_loader.LiberoTaskEpisodeFilteredDataset._extract_task_episode_arrays(dataset)
    selected_indices, owners, episodes_by_task = _select_anchor_indices(
        task_col,
        episode_col,
        episodes_per_task=args.episodes_per_task,
        anchors_per_episode=args.anchors_per_episode,
    )
    dataset = _data_loader.transform_dataset(dataset, data_config)

    logging.info("Loading checkpoint model from %s", checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    action_stats = norm_stats["actions"]
    action_mean = np.asarray(action_stats.mean[:7], dtype=np.float32)
    action_std = np.asarray(action_stats.std[:7], dtype=np.float32) + 1e-6

    rng = jax.random.key(args.seed)
    anchor_metrics: list[dict[str, float | int]] = []
    total = len(selected_indices)
    for batch_start in tqdm.tqdm(range(0, total, args.batch_size), desc="Offline action prediction"):
        real_indices = selected_indices[batch_start : batch_start + args.batch_size]
        real_owners = owners[batch_start : batch_start + args.batch_size]
        real_count = len(real_indices)

        # Keep one static batch shape so JAX compiles sample_actions only once.
        if real_count < args.batch_size:
            pad_count = args.batch_size - real_count
            real_indices = real_indices + [real_indices[-1]] * pad_count
            real_owners = real_owners + [real_owners[-1]] * pad_count

        samples = [dataset[index] for index in real_indices]
        batch = _stack_samples(samples)
        observation = _model.Observation.from_dict(batch)
        target_normalized = np.asarray(batch["actions"][:, :, :7], dtype=np.float32)

        rng, sample_rng = jax.random.split(rng)
        predicted_normalized = np.asarray(
            sample_actions(sample_rng, observation, num_steps=args.diffusion_steps)[:, :, :7],
            dtype=np.float32,
        )

        predicted = predicted_normalized * action_std + action_mean
        target = target_normalized * action_std + action_mean
        error = predicted - target

        for batch_index in range(real_count):
            task_id, episode_id = real_owners[batch_index]
            position_rmse = float(np.sqrt(np.mean(np.square(error[batch_index, :, :3]))))
            rotation_rmse = float(np.sqrt(np.mean(np.square(error[batch_index, :, 3:6]))))
            normalized_rmse = float(
                np.sqrt(np.mean(np.square(predicted_normalized[batch_index] - target_normalized[batch_index])))
            )
            predicted_gripper_open = predicted[batch_index, :, 6] >= 0.05
            target_gripper_open = target[batch_index, :, 6] >= 0.05
            gripper_accuracy = float(np.mean(predicted_gripper_open == target_gripper_open))
            anchor_metrics.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "position_rmse_mm": position_rmse,
                    "rotation_rmse_rad": rotation_rmse,
                    "gripper_accuracy": gripper_accuracy,
                    "normalized_rmse": normalized_rmse,
                }
            )

    grouped: dict[tuple[int, int], list[dict[str, float | int]]] = {}
    for metric in anchor_metrics:
        key = (int(metric["task_id"]), int(metric["episode_id"]))
        grouped.setdefault(key, []).append(metric)

    episode_results: list[dict[str, float | int | bool]] = []
    for (task_id, episode_id), metrics in sorted(grouped.items()):
        result: dict[str, float | int | bool] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "position_rmse_mm": float(np.mean([float(x["position_rmse_mm"]) for x in metrics])),
            "rotation_rmse_rad": float(np.mean([float(x["rotation_rmse_rad"]) for x in metrics])),
            "gripper_accuracy": float(np.mean([float(x["gripper_accuracy"]) for x in metrics])),
            "normalized_rmse": float(np.mean([float(x["normalized_rmse"]) for x in metrics])),
        }
        for threshold_name, threshold in THRESHOLDS.items():
            result[f"{threshold_name}_proxy_success"] = _passes_threshold(result, threshold)  # type: ignore[arg-type]
        episode_results.append(result)

    task_names = {}
    root = pathlib.Path(data_config.root or "")
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        for line in tasks_path.read_text().splitlines():
            row = json.loads(line)
            task_names[int(row["task_index"])] = row["task"]

    task_summary: dict[str, dict] = {}
    for task_id in sorted(episodes_by_task):
        rows = [row for row in episode_results if int(row["task_id"]) == task_id]
        summary = {
            "task_name": task_names.get(task_id, f"task_{task_id}"),
            "episodes": len(rows),
            "position_rmse_mm_mean": float(np.mean([float(x["position_rmse_mm"]) for x in rows])),
            "rotation_rmse_rad_mean": float(np.mean([float(x["rotation_rmse_rad"]) for x in rows])),
            "gripper_accuracy_mean": float(np.mean([float(x["gripper_accuracy"]) for x in rows])),
            "normalized_rmse_mean": float(np.mean([float(x["normalized_rmse"]) for x in rows])),
        }
        for threshold_name in THRESHOLDS:
            success_key = f"{threshold_name}_proxy_success"
            summary[f"{threshold_name}_proxy_success_rate"] = float(np.mean([bool(x[success_key]) for x in rows]))
        task_summary[str(task_id)] = summary

    overall = {"episodes": len(episode_results)}
    for threshold_name in THRESHOLDS:
        success_key = f"{threshold_name}_proxy_success"
        overall[f"{threshold_name}_proxy_success_rate"] = float(
            np.mean([bool(x[success_key]) for x in episode_results])
        )

    report = {
        "evaluation_type": "teacher_forced_offline_action_agreement_proxy",
        "warning": "Proxy success is not real-robot rollout success; evaluation observations come from training demos.",
        "checkpoint_dir": str(checkpoint_dir),
        "config_name": args.config_name,
        "seed": args.seed,
        "anchors_per_episode": args.anchors_per_episode,
        "diffusion_steps": args.diffusion_steps,
        "thresholds": THRESHOLDS,
        "tasks": task_summary,
        "overall": overall,
    }

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    csv_path = output_dir / "episodes.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_results[0].keys()))
        writer.writeheader()
        writer.writerows(episode_results)

    logging.info("Summary written to %s", json_path)
    logging.info("Per-episode results written to %s", csv_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    jax.config.update("jax_debug_infs", False)
    evaluate(tyro.cli(Args))
