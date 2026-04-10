from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env"
GUIX_RUN = ROOT / "scripts" / "guix-run"
KNOWN_HOSTS_FILE = ROOT / ".remote-known-hosts"
ENV_PREFIX = "SLINOSS_LM_REMOTE"
LEGACY_ENV_PREFIX = "KD_REMOTE"


class RemoteConfigError(RuntimeError):
    pass


AuthMode = Literal["key", "password"]


@dataclass(frozen=True)
class RemoteMachine:
    name: str
    host: str
    user: str
    port: int
    workdir: str | None
    auth: AuthMode
    ssh_key: str | None
    password: str | None

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RemoteConfigError(f"missing remote env file: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            raise RemoteConfigError(
                f"invalid env line {line_number} in {path}: {raw_line!r}"
            )
        key = key.strip()
        parsed = shlex.split(value, posix=True)
        if len(parsed) > 1:
            raise RemoteConfigError(
                "env value for "
                f"{key} on line {line_number} must parse to one token, "
                f"got {len(parsed)}"
            )
        values[key] = parsed[0] if parsed else ""
    return values


def load_remote_env(path: Path) -> dict[str, str]:
    values = parse_env_file(path)
    for key, value in os.environ.items():
        if key.startswith(f"{ENV_PREFIX}_") or key.startswith(f"{LEGACY_ENV_PREFIX}_"):
            values[key] = value
    return values


def normalize_machine_name(name: str) -> str:
    return name.upper().replace("-", "_").replace(".", "_")


def configured_machine_names(env: dict[str, str]) -> list[str]:
    configured = env.get(f"{ENV_PREFIX}_MACHINES", "") or env.get(
        f"{LEGACY_ENV_PREFIX}_MACHINES", ""
    )
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]

    return [
        key[len(f"{ENV_PREFIX}_") : -len("_HOST")].lower()
        for key in sorted(env)
        if key.startswith(f"{ENV_PREFIX}_") and key.endswith("_HOST")
    ] or [
        key[len(f"{LEGACY_ENV_PREFIX}_") : -len("_HOST")].lower()
        for key in sorted(env)
        if key.startswith(f"{LEGACY_ENV_PREFIX}_") and key.endswith("_HOST")
    ]


def resolve_machine_name(env: dict[str, str], requested: str | None) -> str:
    if requested:
        return requested
    if env.get(f"{ENV_PREFIX}_MACHINE"):
        return env[f"{ENV_PREFIX}_MACHINE"]
    if env.get(f"{LEGACY_ENV_PREFIX}_MACHINE"):
        return env[f"{LEGACY_ENV_PREFIX}_MACHINE"]
    if env.get(f"{ENV_PREFIX}_DEFAULT_MACHINE"):
        return env[f"{ENV_PREFIX}_DEFAULT_MACHINE"]
    if env.get(f"{LEGACY_ENV_PREFIX}_DEFAULT_MACHINE"):
        return env[f"{LEGACY_ENV_PREFIX}_DEFAULT_MACHINE"]

    names = configured_machine_names(env)
    if len(names) == 1:
        return names[0]

    configured = ", ".join(names) if names else "<none>"
    raise RemoteConfigError(
        f"select a machine with --machine; configured machines: {configured}"
    )


def machine_field(env: dict[str, str], machine: str, field: str) -> str | None:
    normalized = normalize_machine_name(machine)
    return env.get(f"{ENV_PREFIX}_{normalized}_{field}") or env.get(
        f"{LEGACY_ENV_PREFIX}_{normalized}_{field}"
    )


def require_machine_field(env: dict[str, str], machine: str, field: str) -> str:
    value = machine_field(env, machine, field)
    if value is None or value == "":
        raise RemoteConfigError(
            f"missing required field {field} for machine {machine!r}"
        )
    return value


def resolve_machine(env: dict[str, str], requested: str | None) -> RemoteMachine:
    name = resolve_machine_name(env, requested)
    host = require_machine_field(env, name, "HOST")
    user = require_machine_field(env, name, "USER")
    workdir = machine_field(env, name, "WORKDIR") or None
    port_text = machine_field(env, name, "PORT") or "22"
    auth_text = (machine_field(env, name, "AUTH") or "key").lower()
    ssh_key = machine_field(env, name, "SSH_KEY") or None
    password = machine_field(env, name, "PASSWORD") or None

    try:
        port = int(port_text)
    except ValueError as exc:
        raise RemoteConfigError(
            f"invalid port {port_text!r} for machine {name!r}"
        ) from exc

    if ssh_key is not None:
        ssh_key = str(Path(ssh_key).expanduser())

    if auth_text == "key":
        auth: AuthMode = "key"
    elif auth_text == "password":
        auth = "password"
        if not password:
            raise RemoteConfigError(
                f"missing PASSWORD for password-auth machine {name!r}"
            )
    else:
        raise RemoteConfigError(
            f"unsupported auth mode {auth_text!r} for machine {name!r}"
        )

    return RemoteMachine(
        name=name,
        host=host,
        user=user,
        port=port,
        workdir=workdir,
        auth=auth,
        ssh_key=ssh_key,
        password=password,
    )


def prefixed_env(machine: RemoteMachine) -> dict[str, str]:
    env = os.environ.copy()
    if uses_sshpass(machine):
        assert machine.password is not None
        env["SSHPASS"] = machine.password
    return env


def quoted_command(parts: list[str]) -> str:
    return shlex.join(parts)


def uses_sshpass(machine: RemoteMachine) -> bool:
    return machine.password is not None


def ssh_auth_options(machine: RemoteMachine) -> list[str]:
    options: list[str] = []
    if machine.auth == "password":
        return [
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
        ]
    if machine.ssh_key:
        options += ["-i", machine.ssh_key]
    if machine.password is not None:
        options += ["-o", "PreferredAuthentications=publickey,password"]
    else:
        options += ["-o", "BatchMode=yes"]
    return options


def ssh_command(
    machine: RemoteMachine,
    *,
    allocate_tty: bool,
    remote_command: str | None,
) -> list[str]:
    command: list[str] = [str(GUIX_RUN)]
    if uses_sshpass(machine):
        command += ["sshpass", "-e"]
    command.append("ssh")
    if allocate_tty:
        command.append("-t")
    command += [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "VisualHostKey=no",
        "-o",
        "WarnWeakCrypto=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-p",
        str(machine.port),
    ]
    command += ssh_auth_options(machine)
    command.append(machine.target)
    if remote_command is not None:
        command.append(f"bash -lc {shlex.quote(remote_command)}")
    return command


def rsync_ssh_transport(machine: RemoteMachine) -> str:
    parts: list[str] = []
    if uses_sshpass(machine):
        parts += ["sshpass", "-e"]
    parts += [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "VisualHostKey=no",
        "-o",
        "WarnWeakCrypto=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-p",
        str(machine.port),
    ]
    parts += ssh_auth_options(machine)
    return quoted_command(parts)


def render_command(command: list[str], *, machine: RemoteMachine) -> str:
    prefix = "SSHPASS=<redacted> " if uses_sshpass(machine) else ""
    return prefix + quoted_command(command)


def run_command(command: list[str], *, machine: RemoteMachine, dry_run: bool) -> int:
    if dry_run:
        print(render_command(command, machine=machine))
        return 0
    completed = subprocess.run(command, env=prefixed_env(machine), check=False)
    return completed.returncode


def remote_command_for_shell(
    *,
    command: list[str],
    workdir: str | None,
) -> tuple[bool, str | None]:
    if command:
        joined = quoted_command(command)
        if workdir:
            return False, f"cd {shlex.quote(workdir)} && {joined}"
        return False, joined
    if workdir:
        return True, f"cd {shlex.quote(workdir)} && exec ${{SHELL:-/bin/sh}} -l"
    return True, None


def command_shell(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)
    workdir = None if args.no_workdir else (args.cwd or machine.workdir)
    allocate_tty, remote_command = remote_command_for_shell(
        command=args.command,
        workdir=workdir,
    )
    return run_command(
        ssh_command(machine, allocate_tty=allocate_tty, remote_command=remote_command),
        machine=machine,
        dry_run=args.dry_run,
    )


def upload_excludes() -> list[str]:
    return [
        ".env",
        ".env.local",
        ".git/",
        ".mypy_cache/",
        ".nox/",
        ".pytest_cache/",
        ".pyright/",
        ".ruff_cache/",
        ".venv/",
        "__pycache__/",
        "build/",
        "dist/",
        "log/",
        "remote-downloads/",
        "runs/",
        "*.pyc",
    ]


def ensure_remote_directory(machine: RemoteMachine, path: str, *, dry_run: bool) -> int:
    return run_command(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=f"mkdir -p {shlex.quote(path)}",
        ),
        machine=machine,
        dry_run=dry_run,
    )


def command_rsync(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)

    if args.direction == "upload":
        source = args.source or f"{ROOT}/"
        if args.dest is not None:
            dest = args.dest
        elif machine.workdir is not None:
            dest = f"{machine.workdir}/"
        else:
            message = f"machine {machine.name!r} has no WORKDIR; pass --dest explicitly for upload"
            raise RemoteConfigError(message)
    else:
        if args.source is not None:
            source = args.source
        elif machine.workdir is not None:
            source = f"{machine.workdir}/"
        else:
            message = f"machine {machine.name!r} has no WORKDIR; pass --source explicitly for download"
            raise RemoteConfigError(message)
        dest = args.dest or str(ROOT / "remote-downloads" / machine.name) + "/"

    if args.direction == "upload":
        status = ensure_remote_directory(machine, dest, dry_run=args.dry_run)
        if status != 0:
            return status
    else:
        mkdir_command = ["mkdir", "-p", dest]
        if args.dry_run:
            print(quoted_command(mkdir_command))
        else:
            subprocess.run(mkdir_command, check=True)

    command: list[str] = [str(GUIX_RUN)]
    command += ["rsync", "-az"]
    if args.delete:
        command.append("--delete")
    if args.direction == "upload":
        for pattern in upload_excludes():
            command += ["--exclude", pattern]
    command += ["-e", rsync_ssh_transport(machine)]
    if args.direction == "upload":
        command += [source, f"{machine.target}:{dest}"]
    else:
        command += [f"{machine.target}:{source}", dest]
    return run_command(command, machine=machine, dry_run=args.dry_run)


def command_list(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    for name in configured_machine_names(env):
        print(name)
    return 0


def command_print_config(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)
    print(f"env_file={args.env_file}")
    print(f"machine={machine.name}")
    print(f"host={machine.host}")
    print(f"user={machine.user}")
    print(f"port={machine.port}")
    print(f"workdir={machine.workdir or ''}")
    print(f"auth={machine.auth}")
    print(f"ssh_key={machine.ssh_key or ''}")
    print(f"has_password={'true' if machine.password else 'false'}")
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)
    smoke = (
        "set -eu; "
        "echo 'host='$(hostname); "
        "echo 'pwd='$(pwd); "
        "echo 'user='$(whoami); "
        "if command -v python3 >/dev/null 2>&1; then python3 --version; fi; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        "--format=csv,noheader; "
        "fi"
    )
    remote_command = (
        f"cd {shlex.quote(machine.workdir)} && {smoke}" if machine.workdir else smoke
    )
    return run_command(
        ssh_command(machine, allocate_tty=False, remote_command=remote_command),
        machine=machine,
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remotectl")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(
            os.environ.get(
                f"{ENV_PREFIX}_ENV_FILE",
                os.environ.get(f"{LEGACY_ENV_PREFIX}_ENV_FILE", DEFAULT_ENV_FILE),
            )
        ),
        help="Path to the machine env file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured remote machines")
    list_parser.set_defaults(handler=command_list)

    config_parser = subparsers.add_parser(
        "print-config",
        help="Print resolved machine configuration",
    )
    config_parser.add_argument("--machine", help="Machine name from the env file")
    config_parser.set_defaults(handler=command_print_config)

    shell_parser = subparsers.add_parser(
        "shell", help="Open a remote shell or run a command"
    )
    shell_parser.add_argument("--machine", help="Machine name from the env file")
    shell_parser.add_argument("--cwd", help="Override the remote working directory")
    shell_parser.add_argument(
        "--no-workdir",
        action="store_true",
        help="Do not cd into the configured workdir",
    )
    shell_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command instead of executing it",
    )
    shell_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run remotely",
    )
    shell_parser.set_defaults(handler=command_shell)

    rsync_parser = subparsers.add_parser(
        "rsync", help="Sync files to or from a remote machine"
    )
    rsync_parser.add_argument("--machine", help="Machine name from the env file")
    rsync_parser.add_argument(
        "--direction",
        choices=["upload", "download"],
        default="upload",
        help="Whether to upload to the remote or download back",
    )
    rsync_parser.add_argument("--source", help="Override the sync source path")
    rsync_parser.add_argument("--dest", help="Override the sync destination path")
    rsync_parser.add_argument(
        "--delete", action="store_true", help="Pass --delete to rsync"
    )
    rsync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands instead of executing them",
    )
    rsync_parser.set_defaults(handler=command_rsync)

    smoke_parser = subparsers.add_parser(
        "smoke", help="Run a small remote sanity check"
    )
    smoke_parser.add_argument("--machine", help="Machine name from the env file")
    smoke_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command instead of executing it",
    )
    smoke_parser.set_defaults(handler=command_smoke)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = getattr(args, "command", None)
    if isinstance(command, list) and command and command[0] == "--":
        args.command = command[1:]
    try:
        return int(args.handler(args))
    except RemoteConfigError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "error: interrupted\n")


if __name__ == "__main__":
    raise SystemExit(main())
