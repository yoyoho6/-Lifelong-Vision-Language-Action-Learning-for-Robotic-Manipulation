# LifelongVLA for OpenPI

An isolated research branch that brings lifelong Vision-Language-Action learning to the JAX/Flax implementation of [OpenPI](https://github.com/Physical-Intelligence/openpi).

This project is inspired by [*Towards Human-like Physical Intelligence: Lifelong Vision-Language-Action Learning for Robotic Manipulation*](https://arxiv.org/abs/2607.14852) and implements its core ideas for the π0 architecture:

- short-term and long-term LoRA pathways;
- a sample-conditioned gate computed from frozen prefix features;
- compact replay containing prefix features, robot states, and actions only;
- fresh diffusion-time and noise sampling at every replay step;
- suffix reconstruction with the current model;
- task-boundary teacher distillation;
- resumable model, teacher, and replay-buffer checkpoints.

This repository is an independent, paper-inspired implementation. It is not the authors' official code and does not claim exact reproduction of the reported results. The implementation is kept separate from OpenPI's default `scripts/train.py` and `openpi.models.pi0.Pi0` paths.

## Method

Each adapted attention or feed-forward weight contains two low-rank residuals:

```text
W = W0 + (1 - alpha(c)) * A_short B_short + alpha(c) * A_long B_long
```

The gate context `c` is obtained by masked mean pooling over stop-gradient prefix tokens. The two pathways use different supervision and update speeds:

- the current-task denoising loss updates the short-term LoRA pathway;
- replay and teacher-distillation losses update the long-term LoRA pathway;
- the shared gate receives both current and replay supervision;
- long-term gradients are multiplied by `--long-lr-scale` to provide slower consolidation.

The optimization objective is:

```text
L = L_new + replay_weight * L_replay + distill_weight * L_distill
```

Current and replay sub-batches are normalized separately and combined in one optimizer update.

## Cache-efficient stochastic replay

Each replay entry contains exactly:

```text
prefix_tokens
prefix_mask
state
actions
```

The buffer does not store raw images, language-token IDs, suffix tokens, diffusion times, or diffusion noise. For every replay update, the trainer samples a fresh diffusion time from `Beta(1.5, 1)` and fresh Gaussian noise, then reconstructs the action suffix using the current model.

Prefix features are persisted as `float16`. This has the same two-byte storage footprint as `bfloat16` while remaining portable through NumPy NPZ serialization.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── LICENSES/Apache-2.0.txt
├── .gitignore
├── pyproject.toml
├── uv.lock
├── assets/                    # Normalization statistics
├── docs/                      # OpenPI runtime documentation
├── examples/                  # Robot clients and dataset conversion
├── packages/openpi-client/    # WebSocket policy client
├── scripts/
│   ├── train.py               # Standard OpenPI trainer
│   ├── serve_policy.py        # Standard OpenPI policy server
│   └── lifelong_vla/
│       ├── train.py           # LifelongVLA trainer
│       └── serve_policy.py    # LifelongVLA policy server
├── src/openpi/                # Complete OpenPI Python package
│   └── lifelong_vla/
│       ├── __init__.py
│       ├── config.py
│       ├── dual_lora.py
│       ├── gemma.py
│       ├── model.py
│       ├── replay_buffer.py
│       ├── dual_lora_test.py
│       └── replay_buffer_test.py
└── third_party/libero/        # Bundled LIBERO Python package
```

## Installation

This repository contains the OpenPI runtime and can be installed directly. It requires Python 3.11 and is configured for JAX with CUDA 12.

```bash
git clone https://github.com/yoyoho6/-Lifelong-Vision-Language-Action-Learning-for-Robotic-Manipulation.git
cd ./-Lifelong-Vision-Language-Action-Learning-for-Robotic-Manipulation
uv sync
```

For LIBERO simulation and evaluation, install the bundled package into the same environment:

```bash
uv pip install -e third_party/libero
```

Robot datasets and pretrained checkpoints are intentionally not committed. Configure their paths in `src/openpi/training/config.py`, or select an existing configuration whose paths are valid on your machine. The trainer reuses that π0 configuration for its dataset, normalization statistics, optimizer, and pretrained checkpoint while replacing the model with `LifelongPi0Config`.

## Dataset requirements

The selected OpenPI base configuration must:

1. use `Pi0Config`;
2. use a data factory exposing `libero_task_indices` and `libero_replay_episodes`;
3. support selecting one task with `libero_task_indices=[task_id]`;
4. transform robot states and actions to the dimensions expected by π0.

The bundled `LeRobotXarmDataConfig` and `LeRobotLiberoDataConfig` classes satisfy this interface. During lifelong training, each task loader reads only the current task. Previous raw trajectories are never reloaded for replay.

## Training

The following example uses the existing `pi0_xarm_data_heyao_2_incremental_lora` OpenPI configuration:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/lifelong_vla/train.py \
  --base-config pi0_xarm_data_heyao_2_incremental_lora \
  --exp-name lifelong_vla_xarm_5tasks \
  --task-sequence 0 1 2 3 4 \
  --steps-per-task 10000 \
  --batch-size 4 \
  --short-rank 16 \
  --long-rank 16 \
  --lora-alpha 16 \
  --replay-capacity 500 \
  --cache-per-task 50 \
  --replay-batch-size 4 \
  --replay-weight 1.0 \
  --distill-weight 0.1 \
  --long-lr-scale 0.1
```

If the base configuration already defines `libero_task_sequence` and `steps_per_task`, those arguments may be omitted. The paper's principal settings are 10,000 updates per task, LoRA rank 16, a total cache capacity of 500, 50 cached samples per task, replay weight 1.0, and distillation weight 0.1.

View all options with:

```bash
uv run scripts/lifelong_vla/train.py --help
```

## Checkpoints and resume

Model parameters continue to use OpenPI's Orbax checkpoint format. Additional lifelong-learning state is stored as follows:

- `teacher_params` is included in the training state;
- the feature buffer is written to `<checkpoint_root>/lifelong_replay/<step>.npz`;
- task IDs, per-task sample counts, and buffer RNG state are saved in the replay sidecar.

Resume with the same model-structure arguments used for the initial run, especially both LoRA ranks:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/lifelong_vla/train.py \
  --base-config pi0_xarm_data_heyao_2_incremental_lora \
  --exp-name lifelong_vla_xarm_5tasks \
  --task-sequence 0 1 2 3 4 \
  --steps-per-task 10000 \
  --short-rank 16 \
  --long-rank 16 \
  --resume
```

`--resume` and `--overwrite` are mutually exclusive. If the replay sidecar matching the restored checkpoint is absent, training stops instead of silently continuing without replay.

## Inference server

Task identity is not required during inference. The gate derives the short/long mixture directly from the current observation's prefix features:

```bash
uv run scripts/lifelong_vla/serve_policy.py \
  --base-config pi0_xarm_data_heyao_2_incremental_lora \
  --checkpoint-dir checkpoints/pi0_xarm_data_heyao_2_incremental_lora/lifelong_vla_xarm_5tasks/50000 \
  --short-rank 16 \
  --long-rank 16 \
  --port 8000
```

The values of `--short-rank`, `--long-rank`, `--lora-alpha`, and `--gate-bias` must match training.

## Tests

From the repository root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest -q src/openpi/lifelong_vla

uv run ruff check src/openpi/lifelong_vla scripts/lifelong_vla
```

The tests cover:

- dual-LoRA broadcasting when the batch axis is not the leading axis;
- short/long gradient isolation;
- a tiny dual-LoRA Gemma forward pass;
- per-task replay quotas and task-excluded sampling;
- replay-buffer persistence, including `bfloat16` input conversion.

## Design boundaries

- Feature replay requires a frozen prefix encoder. This branch therefore freezes every parameter except dual-LoRA factors and gate parameters. Unfreezing the visual or language prefix encoder would make cached representations stale.
- Replay NPZ files are uncompressed to reduce checkpoint latency. A cached sample is dominated by `prefix_length × hidden_width × dtype_bytes`; adjust `--cache-per-task` for the available disk budget and number of camera views.
- When the buffer is full, bounded random replacement is used. The default `10 tasks × 50 samples = 500` exactly matches the default capacity.
- The current and replay samples are evaluated as separate sub-batches in the same gradient computation. Their separately normalized losses are mathematically equivalent to the paper's masked objective for this model, while avoiding fabricated image observations for cached features.

## Citation

```bibtex
@misc{he2026humanlikephysicalintelligencelifelongvisionlanguageaction,
  title={Towards Human-like Physical Intelligence: Lifelong Vision-Language-Action Learning for Robotic Manipulation},
  author={Yao He and Gan Sun and Wenqi Liang and Fazeng Li and Yang Cong},
  year={2026},
  eprint={2607.14852},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## License

The repository-level license is MIT. Files adapted from OpenPI retain their original Apache-2.0 headers and remain subject to the corresponding upstream license terms.
