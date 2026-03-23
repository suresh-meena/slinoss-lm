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

