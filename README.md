# SLinOSS LM

`slinoss-lm` is a single-node training and evaluation harness for SLinOSS language-model runs on packed FineWeb-Edu data.

The repo is intentionally narrow:

- plain `torchrun` single-node training
- no Slurm, ZeRO, FSDP, or cluster scheduler integration
- prepacked `int32` token rows, not raw parquet or on-the-fly tokenization
- local or user-managed runtime overlays, not machine-specific configs checked into git

## Design Choices

The harness uses plain PyTorch DDP, not FSDP or ZeRO.

That is deliberate:

- DDP is simpler to debug, more stable under interruptions, and easier to checkpoint and resume cleanly
- the main bottlenecks at these scales are throughput and operational reliability, not parameter sharding
- hardware selection and runtime tuning are intentionally left to local overlays instead of being baked into the repo

The data loader is fixed-order and deterministic:

- it reads the transferred packed corpus from `FWEDU_DATA_ROOT/data/fwedu-llama31-2k`
- it never re-tokenizes, re-packs, or reorders tokens inside a row
- each global batch covers a contiguous window of already-packed sequences
- DDP ranks receive disjoint contiguous slices of that global batch

The checkpointing story is also conservative:

- rank 0 writes atomic checkpoint directories
- the latest checkpoint is tracked by a JSON marker, not a symlink
- `SIGINT` and `SIGTERM` trigger a clean stop after the current optimizer step and force an emergency checkpoint
- resuming picks up model, optimizer, scheduler, RNG state, and consumed-token counters

## Dataset Contract

The training environment is expected to have the transferred `fwedu-data` package in place.

Set:

```bash
export FWEDU_DATA_ROOT=/path/to/fwedu-data
```

The harness expects the packed training set at:

```text
$FWEDU_DATA_ROOT/data/fwedu-llama31-2k
```

This path is the exact output of the `fineweb` transfer workflow.

## Experiments Shipped Here

Four parameter-matched SLinOSS runs are configured:

- `fwedu-180m`: `14` layers, `d_model=512`, `intermediate_size=1536`
- `fwedu-440m`: `18` layers, `d_model=768`, `intermediate_size=2560`
- `fwedu-880m`: `22` layers, `d_model=1024`, `intermediate_size=3328`
- `fwedu-1p5b`: `26` layers, `d_model=1280`, `intermediate_size=3840`

Shared mixer defaults:

- `d_state=128`
- `expand=2`
- `d_head=64`
- `d_conv=4`
- `chunk_size=64`
- `dt_min=1e-3`
- `dt_init_floor=1e-3`
- `r_min=0.2`

Training contract:

- tokenizer family: `Llama-3.1`
- vocab size: `128256`
- context length: `2048`
- target training budget: `100,000,000,000` tokens
- precision: `bf16`
- optimizer: AdamW
- betas: `(0.9, 0.95)`
- weight decay: `0.1`
- grad clip: `1.0`
- dropout: `0.0`
- LR schedule: linear warmup, then cosine decay to `1e-5`

Peak LR choices used here:

- `180M`: `4e-4`
- `440M`: `3e-4`

Those values are fixed per scale and should be held constant across architecture baselines at the same scale.

## Install

Use Python `3.11`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the base harness plus the local dev tools used by the repo. The published `slinoss` wheel declared in `pyproject.toml` is used, not a sibling editable checkout.

For zero-shot evaluation, install the optional eval stack in the same `.venv`:

```bash
source .venv/bin/activate
pip install -r requirements-eval.txt
```

For W&B logging, install the optional tracker dependency in the same `.venv`:

```bash
source .venv/bin/activate
pip install -r requirements-wandb.txt
```

## Inspect Before Launching

Use the inspector to verify the resolved config, parameter count, and batch geometry.

Pass `--world-size` when you want the reported batch geometry to match a planned DDP launch.

```bash
source .venv/bin/activate
python3 -m slinoss_lm.inspect --world-size 2 --config configs/experiments/fwedu-180m.yaml
python3 -m slinoss_lm.inspect --world-size 2 --config configs/experiments/fwedu-880m.yaml
```

## Launch Commands

Example:

```bash
source .venv/bin/activate
export FWEDU_DATA_ROOT=/path/to/fwedu-data
torchrun --standalone --nproc-per-node=2 train.py \
  --config configs/experiments/fwedu-180m.yaml
```

If you need host-specific batch geometry, keep that in an untracked local config and pass it as an additional `--config`.

Wall-clock aware stopping is supported through config or environment overrides.
This is useful for queue systems where jobs must checkpoint before hard timeout:

```bash
export SLINOSS_WALL_CLOCK_DEADLINE_UNIX=<unix-seconds>
export SLINOSS_WALL_CLOCK_EXIT_MARGIN_SECONDS=900
```

Equivalent config fields are:

- `train.wall_clock_deadline_unix`
- `train.max_runtime_seconds`
- `train.wall_clock_exit_margin_seconds`

## Outputs

Each run creates a directory under `runs/`:

```text
runs/<run-name>/
├── checkpoints/
│   ├── latest.json
│   └── step-000005000/
│       ├── meta.json
│       └── trainer.pt
├── metrics.jsonl
├── resolved-config.yaml
├── run-state.json
└── system-info.json
```

When W&B is enabled, the run directory also contains:

```text
runs/<run-name>/wandb-run.json
```

`trainer.pt` contains:

- model state
- optimizer state
- scheduler state
- RNG state
- consumed-token counters
- step counters

To resume, rerun the same launch command with `--resume auto` or leave the default unchanged. The trainer will detect `checkpoints/latest.json` and continue from there.

## Evaluation

Two evaluation entrypoints are included.

### 1. Zero-shot downstream evaluation

This exports the selected checkpoint as a Hugging Face-compatible local model and runs `lm-eval` against the standard zero-shot suite from the fact sheet:

- `lambada_openai`
- `hellaswag`
- `piqa`
- `arc_easy`
- `arc_challenge`
- `winogrande`
- `openbookqa`

Example:

```bash
source .venv/bin/activate
export LLAMA31_TOKENIZER=/path/to/local/llama31-tokenizer
python3 eval_zero_shot.py \
  --config configs/experiments/fwedu-180m.yaml \
  --checkpoint runs/fwedu-180m/checkpoints/step-000190000/trainer.pt
```

The tokenizer must be available locally or through Hugging Face auth. Training itself does not require tokenizer access because the corpus is already tokenized.

### 2. Packed-set perplexity evaluation

This evaluates a checkpoint on any packed dataset range you specify. That is useful for:

- a separately prepared held-out packed slice
- a post-hoc validation corpus you choose to transfer later

Example:

```bash
source .venv/bin/activate
python3 eval_ppl.py \
  --config configs/experiments/fwedu-180m.yaml \
  --checkpoint runs/fwedu-180m/checkpoints/step-000190000/trainer.pt \
  --dataset-root /path/to/packed-eval-root \
  --batch-size 8
```

## Operational Notes

- Keep training and evaluation in separate processes.
- Do not mutate the packed corpus between resumes.
- Do not change world size, microbatch size, or gradient-accumulation settings mid-run unless you are intentionally starting a new run.
- This repo saves full-state checkpoints, so disaster recovery is straightforward but storage use is real. Keep an eye on disk.
- `torch.compile` is disabled by default because the primary goal here is reliable paper-grade runs. You can enable it in a runtime config once a scale has proven stable.

## W&B

W&B support is optional and machine-independent:

- it is controlled by the `wandb` config section
- only rank `0` logs in DDP
- local `metrics.jsonl` and `run-state.json` remain the source-of-truth fallback
- authentication is handled through standard W&B environment variables such as `WANDB_API_KEY`
- online and offline modes are both supported through `wandb.mode`

Example config fragment:

```yaml
wandb:
  enabled: true
  project: slinoss-pretrain
  entity: null
  group: fw-100b
  run_name: null
  run_id: null
  tags: [slinoss, fwedu, 180m]
  mode: online
  resume: allow
  upload_checkpoints: false
```

If a run resumes from the same run directory, the trainer reuses the previously
persisted W&B run id from `wandb-run.json`.

## What Is Not Assumed

This repo does not assume:

- Slurm
- DeepSpeed
- FSDP
- ZeRO
- raw FineWeb parquet access on the training servers

It assumes only:

- the transferred packed `fwedu-data` bundle
- Python `3.11`
- `torchrun`
