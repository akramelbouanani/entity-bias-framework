#!/usr/bin/env python3
"""Run configured generalization audits against a completion API."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from bias_audit.config import DEFAULT_CONFIG_DIR, PROJECT_ROOT, ConfigError, TaskRegistry
from bias_audit.data import expand_task
from bias_audit.inference import CompletionClient, output_path, run_dataframe
from bias_audit.server import model_output_name


def boolean(value: str) -> bool:
    normalized = value.casefold()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--few-shot", "--few_shot", dest="few_shot", nargs="+", type=boolean, required=True)
    parser.add_argument("--numerical", nargs="+", type=boolean, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--model", help="Optional override when the server exposes multiple models")
    parser.add_argument("--languages", nargs="+", default=["english"])
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generalization")
    parser.add_argument("--request-batch-size", type=int, default=1000)
    parser.add_argument("--flush-rows", type=int, default=8000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    registry = TaskRegistry(args.config_dir)
    registry.validate(check_data=True)
    client = CompletionClient(
        model=args.model,
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    model_name = model_output_name(client.model)
    print(f"Using server model: {client.model}")

    for task_id in args.tasks:
        task = registry.task(task_id)
        if task.suite != "generalization":
            raise ConfigError(f"Task '{task_id}' belongs to suite '{task.suite}', not 'generalization'")
        entity_set = registry.entity_set(task.entity_set)
        for language in args.languages:
            frame = expand_task(task, entity_set, language=language, variant="main")
            for few_shot, numerical in itertools.product(args.few_shot, args.numerical):
                destination = output_path(
                    args.output_dir,
                    task,
                    language,
                    model_name,
                    few_shot,
                    numerical,
                    "main",
                )
                run_dataframe(
                    frame,
                    task,
                    client,
                    destination,
                    language,
                    few_shot,
                    numerical,
                    args.request_batch_size,
                    args.flush_rows,
                )


if __name__ == "__main__":
    main()
