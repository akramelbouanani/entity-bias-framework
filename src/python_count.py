#!/usr/bin/env python3
"""Verify completeness of configured generalization output files."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from bias_audit.config import DEFAULT_CONFIG_DIR, PROJECT_ROOT, TaskRegistry
from bias_audit.inference import output_path
from run_generalization import boolean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--few-shot", "--few_shot", dest="few_shot", nargs="+", type=boolean, required=True)
    parser.add_argument("--numerical", nargs="+", type=boolean, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--languages", "--language", dest="languages", nargs="+", required=True)
    parser.add_argument("--threshold", type=int, default=800000)
    parser.add_argument("--countries-threshold", type=int, default=200000)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generalization")
    args = parser.parse_args()

    registry = TaskRegistry(args.config_dir)
    failures = []
    for task_id in args.tasks:
        task = registry.task(task_id)
        threshold = args.countries_threshold if task.entity_set == "countries" else args.threshold
        for model, language, few_shot, numerical in itertools.product(
            args.models, args.languages, args.few_shot, args.numerical
        ):
            path = output_path(args.output_dir, task, language, model, few_shot, numerical, "main")
            if not path.exists():
                failures.append((task_id, model, language, few_shot, numerical, "missing"))
                continue
            count = len(pd.read_csv(path).dropna())
            status = "OK" if count >= threshold else "FAILED"
            print(status, task_id, model, language, few_shot, numerical, count)
            if status == "FAILED":
                failures.append((task_id, model, language, few_shot, numerical, count))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

