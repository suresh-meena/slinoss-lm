# Remote Scripts

This directory contains small local helpers for working with remote machines
from this repository.

## Files

- `manifest.scm`
  Guix manifest for the small set of command-line tools used by the remote
  helpers
- `guix-run`
  Run a command inside the Guix shell described by `manifest.scm`
- `remote-list`
  List configured machine names from the root `.env`
- `remote-print-config`
  Print the resolved machine configuration without secrets
- `remote-shell`
  Open an interactive shell or run a command on a selected machine
- `remote-rsync`
  Sync the repository to a machine or pull files back
- `remote-smoke`
  Run a quick connectivity and environment sanity check on a machine

## Local Configuration

Copy `.env.example` to `.env` and fill in the machines you want to manage.

The scripts read:

- `KD_REMOTE_MACHINES`
- `KD_REMOTE_DEFAULT_MACHINE`
- per-machine fields such as:
  `KD_REMOTE_<MACHINE>_HOST`
  `KD_REMOTE_<MACHINE>_USER`
  `KD_REMOTE_<MACHINE>_PORT`
  `KD_REMOTE_<MACHINE>_WORKDIR`
  `KD_REMOTE_<MACHINE>_AUTH`
  `KD_REMOTE_<MACHINE>_SSH_KEY`
  `KD_REMOTE_<MACHINE>_PASSWORD`

Machine names are normalized for the variable prefix by upper-casing and
replacing `-` or `.` with `_`.

`WORKDIR` is optional, but recommended. If it is omitted, `remote-shell` will
use the machine's default login directory and `remote-rsync` will require
explicit `--source` or `--dest` paths instead of defaulting to a workdir.

Auth behavior is intentionally simple:

- `AUTH=key` prefers SSH public-key auth and, when `PASSWORD` is also present,
  falls back to password auth through `sshpass`
- `AUTH=password` forces password auth

That lets the repo keep password access as a recovery path while steadily
moving machines over to public-key auth.

Example:

- machine name: `dgx1`
- env prefix: `KD_REMOTE_DGX1_`

## Examples

```bash
./scripts/remote-list
./scripts/remote-print-config --machine dgx1
./scripts/remote-shell --machine dgx1
./scripts/remote-shell --machine dgx1 -- hostname
./scripts/remote-rsync --machine dgx1
./scripts/remote-rsync --machine dgx1 --direction download --source /srv/kdrifting/runs/ --dest runs/remote-dgx1/
./scripts/remote-smoke --machine dgx1
```

`remote-shell` and `remote-rsync` support `--dry-run` for inspecting the
resolved SSH and rsync commands before executing them.
