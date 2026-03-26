from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "guix-run",
    ROOT / "scripts" / "remote-list",
    ROOT / "scripts" / "remote-print-config",
    ROOT / "scripts" / "remote-smoke",
    ROOT / "scripts" / "remote-shell",
    ROOT / "scripts" / "remote-rsync",
]


def _write_remote_env(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent(
            """
            KD_REMOTE_MACHINES=dgx1,scratch-box
            KD_REMOTE_DEFAULT_MACHINE=dgx1

            KD_REMOTE_DGX1_HOST=dgx.example.edu
            KD_REMOTE_DGX1_USER=alice
            KD_REMOTE_DGX1_PORT=2222
            KD_REMOTE_DGX1_WORKDIR=/srv/kdrifting
            KD_REMOTE_DGX1_AUTH=key
            KD_REMOTE_DGX1_SSH_KEY=/tmp/test-key
            KD_REMOTE_DGX1_PASSWORD=fallback-secret

            KD_REMOTE_SCRATCH_BOX_HOST=10.0.0.5
            KD_REMOTE_SCRATCH_BOX_USER=bob
            KD_REMOTE_SCRATCH_BOX_WORKDIR=/scratch/kdrifting
            KD_REMOTE_SCRATCH_BOX_AUTH=password
            KD_REMOTE_SCRATCH_BOX_PASSWORD=super-secret
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return env_file


def _run_script(
    script_name: str, *args: str, env_file: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["KD_REMOTE_ENV_FILE"] = str(env_file)
    return subprocess.run(
        [str(ROOT / "scripts" / script_name), *args],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )


def test_remote_scripts_have_valid_shell_syntax() -> None:
    for script in SCRIPTS:
        subprocess.run(["sh", "-n", str(script)], check=True, cwd=ROOT)


def test_remote_list_reads_configured_machine_names(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script("remote-list", env_file=env_file)
    assert result.stdout.strip().splitlines() == ["dgx1", "scratch-box"]


def test_remote_print_config_resolves_selected_machine(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-print-config", "--machine", "scratch-box", env_file=env_file
    )
    assert "machine=scratch-box" in result.stdout
    assert "host=10.0.0.5" in result.stdout
    assert "user=bob" in result.stdout
    assert "workdir=/scratch/kdrifting" in result.stdout
    assert "auth=password" in result.stdout
    assert "has_password=true" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_shell_dry_run_uses_resolved_machine_settings(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-shell",
        "--machine",
        "dgx1",
        "--dry-run",
        "--",
        "python",
        "-V",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "ssh" in result.stdout
    assert "/tmp/test-key" in result.stdout
    assert "alice@dgx.example.edu" in result.stdout
    assert "PreferredAuthentications=publickey,password" in result.stdout
    assert "cd /srv/kdrifting && python -V" in result.stdout
    assert "fallback-secret" not in result.stdout


def test_remote_rsync_dry_run_redacts_password_auth(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-rsync",
        "--machine",
        "scratch-box",
        "--dry-run",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "rsync" in result.stdout
    assert "bob@10.0.0.5:/scratch/kdrifting/" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_smoke_dry_run_uses_machine_workdir(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-smoke", "--machine", "dgx1", "--dry-run", env_file=env_file
    )
    assert "alice@dgx.example.edu" in result.stdout
    assert "cd /srv/kdrifting" in result.stdout
    assert "nvidia-smi" in result.stdout
