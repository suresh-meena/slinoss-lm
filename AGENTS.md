# AGENTS.md

## Environment

This repo uses a local Python virtual environment at `.venv`.

- Prefer `source .venv/bin/activate` before running repo-local tooling.
- Prefer `python`, `ruff`, `pyright`, and `pytest` from that activated environment.
- `pyproject.toml` and `requirements.txt` are the pip-facing install surface.
- Keep the install surface aligned with the actual runtime, eval, and dev needs of the repo.

## Commit Conventions

Use lightweight Conventional Commits for all new commits:

- `feat:` new capabilities or APIs
- `fix:` correctness fixes
- `refactor:` structure-preserving code changes
- `perf:` measurable performance work
- `test:` test-only changes
- `docs:` documentation-only changes
- `chore:` repo maintenance that does not affect behavior

Keep commit subjects short and specific.

### Pre-Commit Gate

Before creating any commit, clear this full verification gate in the local `.venv`:

- `ruff format --check .`
- `ruff check`
- `pyright`
- `pytest`

Do not commit while any of these are failing.

## Training Harness Discipline

- Keep the repo focused on single-node language-model experiments.
- Do not add Slurm-, ZeRO-, or FSDP-specific behavior unless there is a concrete need for it.
- Preserve the packed-dataset contract: the loader consumes already-tokenized fixed-length rows and must not silently re-tokenize, re-pack, or inject special tokens.
- Checkpointing and resume behavior are part of the experiment contract. Treat regressions there as correctness bugs.
- Evaluation code should stay compatible with local checkpoint exports and the pinned tokenizer/data contract used by the training runs.

## Remote Control

- This repo ships remote experiment helpers under `scripts/`.
- Machine definitions live in a root `.env` file that must stay out of git.
- Prefer `AUTH=key` with `SSH_KEY` configured. Keep `PASSWORD` populated as a
  fallback while public-key access is being rolled out or repaired.
- Primary commands:
  - `./scripts/remote-list`
  - `./scripts/remote-print-config --machine <name>`
  - `./scripts/remote-shell --machine <name>`
  - `./scripts/remote-rsync --machine <name>`
  - `./scripts/remote-smoke --machine <name>`
- The scripts use a repo-local `.remote-known-hosts` file for non-interactive
  access. Manual aliases like `ssh ampere` or `ssh volta` are managed through
  the user's `~/.ssh/config`.
- Standard remote workflow:
  1. run `./scripts/remote-smoke --machine <name>`
  2. sync the repo with `./scripts/remote-rsync --machine <name>`
  3. launch the experiment through `./scripts/remote-shell --machine <name> -- ...`
- Keep this tooling lean. The goal is reliable remote experiment control, not a
  new orchestration layer.
