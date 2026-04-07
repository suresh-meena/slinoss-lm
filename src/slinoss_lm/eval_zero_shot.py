from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from .checkpoint import load_checkpoint
from .common import atomic_write_json
from .config import load_config
from .configuration_slinoss_lm import SLinOSSLMConfig
from .modeling_slinoss_lm import SLinOSSCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _export_checkpoint(
    *,
    checkpoint_path: Path,
    export_dir: Path,
    config,
    tokenizer_ref: str,
) -> None:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    model = SLinOSSCausalLM(SLinOSSLMConfig(**config.model.architecture_kwargs()))
    model.load_state_dict(payload["model"])
    model.eval()
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_dir, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
    tokenizer.save_pretrained(export_dir)


def _mean_accuracy(results: dict[str, object], tasks: list[str]) -> float | None:
    values: list[float] = []
    result_map = results.get("results", {})
    if not isinstance(result_map, dict):
        return None
    for task in tasks:
        task_metrics = result_map.get(task, {})
        if not isinstance(task_metrics, dict):
            continue
        for key in ("acc_norm,none", "acc,none", "exact_match,none"):
            metric = task_metrics.get(key)
            if isinstance(metric, (int, float)):
                values.append(float(metric))
                break
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    args = build_parser().parse_args()
    if importlib.util.find_spec("lm_eval") is None:
        raise RuntimeError(
            "Zero-shot evaluation requires the optional eval dependencies. "
            "Install them with `pip install -r requirements-eval.txt`."
        )
    config = load_config(args.config, args.overrides)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_dir = checkpoint_path.parent
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_dir / "eval-zero-shot"
    )
    export_dir = output_dir / "hf-export"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_ref = (
        args.tokenizer
        or config.data.tokenizer_path
        or os.environ.get("LLAMA31_TOKENIZER")
        or config.data.tokenizer_name
    )
    if not tokenizer_ref:
        raise RuntimeError(
            "Provide --tokenizer, LLAMA31_TOKENIZER, or data.tokenizer_name."
        )

    _export_checkpoint(
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
        config=config,
        tokenizer_ref=tokenizer_ref,
    )

    batch_size = args.batch_size or config.eval.zero_shot_batch_size
    tasks = ",".join(config.eval.zero_shot_tasks)
    results_path = output_dir / "lm_eval_results.json"
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={export_dir},trust_remote_code=True,dtype=bfloat16,tokenizer={export_dir}",
        "--tasks",
        tasks,
        "--device",
        args.device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(results_path),
    ]
    subprocess.run(cmd, check=True)
    raw = json.loads(results_path.read_text())
    summary = {
        "checkpoint": str(checkpoint_path),
        "export_dir": str(export_dir),
        "tasks": config.eval.zero_shot_tasks,
        "mean_accuracy": _mean_accuracy(raw, config.eval.zero_shot_tasks),
        "results_path": str(results_path),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
