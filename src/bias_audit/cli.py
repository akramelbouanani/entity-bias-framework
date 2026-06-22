"""Command-line tools for inspecting and preparing audits."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT, ConfigError, TaskRegistry
from .data import expand_task
from .prompts import build_prompt
from .server import list_served_models, model_output_name


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bias-audit", description="Configuration-driven entity bias auditing")
    parser.add_argument("--config-dir", type=Path, default=None, help="Override the configs directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered tasks and entity sets")
    list_parser.add_argument("--suite", choices=["generalization", "validation"])

    subparsers.add_parser("validate", help="Validate configuration and CSV schemas")

    server_parser = subparsers.add_parser("server-info", help="Show models exposed by an inference server")
    server_parser.add_argument("--host", default="localhost")
    server_parser.add_argument("--port", type=int, required=True)
    server_parser.add_argument("--timeout", type=float, default=10.0)

    prompt_parser = subparsers.add_parser("prompt", help="Render a prompt for inspection")
    prompt_parser.add_argument("task")
    prompt_parser.add_argument("sentence")
    prompt_parser.add_argument("--language", default="english")
    prompt_parser.add_argument("--target")
    prompt_parser.add_argument("--few-shot", action="store_true")
    prompt_parser.add_argument("--numerical", action="store_true")

    expand_parser = subparsers.add_parser("expand", help="Expand a task over its entity set")
    expand_parser.add_argument("task")
    expand_parser.add_argument("--language", default="english")
    expand_parser.add_argument("--variant", default="main")
    expand_parser.add_argument("--output", type=Path, required=True)

    generation_prompt_parser = subparsers.add_parser(
        "generation-prompt", help="Render one configured sentence-generation prompt"
    )
    generation_prompt_parser.add_argument("task")
    generation_prompt_parser.add_argument("--label", help="Stable label id; defaults to the first label")
    generation_prompt_parser.add_argument("--seed", type=int, default=42)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate a balanced evidence-light dataset through vLLM"
    )
    generate_parser.add_argument("task")
    generate_parser.add_argument("--model", help="Optional override when the server exposes multiple models")
    generate_parser.add_argument("--host", default="localhost")
    generate_parser.add_argument("--port", type=int, required=True)
    generate_parser.add_argument("--output", type=Path, help="Defaults to the task's main dataset")
    generate_parser.add_argument("--checkpoint", type=Path)
    generate_parser.add_argument("--languages", nargs="+", help="Defaults to every configured language")
    generate_parser.add_argument("--samples-per-label", type=int)
    generate_parser.add_argument("--batch-size", type=int, default=32)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--validation-retries", type=int, default=3)
    generate_parser.add_argument("--request-retries", type=int, default=3)
    generate_parser.add_argument("--workers", type=int, default=8, help="Concurrent vLLM chat requests")
    generate_parser.add_argument("--timeout", type=float, default=300.0)
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.add_argument("--allow-partial", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Generate figures from processed audit outputs"
    )
    analyze_subparsers = analyze_parser.add_subparsers(dest="analysis_command", required=True)

    ellipses_parser = analyze_subparsers.add_parser(
        "ellipses", help="Plot bias means and confidence ellipses"
    )
    ellipses_parser.add_argument("entity_set")
    _add_analysis_io_arguments(ellipses_parser)
    ellipses_parser.add_argument(
        "--dimensions", nargs="+", default=["language", "model", "setting"],
        choices=["language", "model", "setting", "task"],
    )
    ellipses_parser.add_argument("--confidence", type=float, default=0.95)
    ellipses_parser.add_argument("--center", action="store_true")

    radar_parser = analyze_subparsers.add_parser(
        "radar", help="Plot task-level bias profiles across entity categories"
    )
    _add_analysis_io_arguments(radar_parser, include_category=False)
    radar_parser.add_argument("--entity-sets", nargs="+")

    boxplots_parser = analyze_subparsers.add_parser(
        "boxplots", help="Plot per-task entity score or F1 distributions"
    )
    boxplots_parser.add_argument("entity_set")
    _add_analysis_io_arguments(boxplots_parser)
    boxplots_parser.add_argument("--metric", choices=["score", "f1"], default="score")
    centering = boxplots_parser.add_mutually_exclusive_group()
    centering.add_argument("--center", dest="center", action="store_true")
    centering.add_argument("--no-center", dest="center", action="store_false")
    boxplots_parser.set_defaults(center=None)
    return parser


def _add_analysis_io_arguments(
    parser: argparse.ArgumentParser, *, include_category: bool = True
) -> None:
    parser.add_argument(
        "--processed-outputs-dir", type=Path, default=PROJECT_ROOT / "processed_outputs"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--models", nargs="+",
        help="Model order for legacy processed files without a .meta.json sidecar",
    )
    if include_category:
        parser.add_argument("--category-column")
        parser.add_argument("--category-order", nargs="+")
    parser.add_argument("--dpi", type=int, default=200)


def main() -> None:
    args = make_parser().parse_args()
    registry = TaskRegistry(config_dir=args.config_dir) if args.config_dir else TaskRegistry()

    if args.command == "list":
        for task in registry.tasks(suite=args.suite):
            print(f"{task.id:18} {task.suite:14} entity_set={task.entity_set} languages={','.join(task.languages)}")
        return
    if args.command == "validate":
        registry.validate(check_data=True)
        from .plotting import load_analysis_spec, validate_analysis_spec

        analysis_path = registry.config_dir / "analysis.json"
        if analysis_path.exists():
            validate_analysis_spec(load_analysis_spec(registry.config_dir), registry)
        generation_tasks = sum(task.generation is not None for task in registry.tasks())
        print(
            f"Configuration is valid: {len(registry.tasks())} tasks, "
            f"{len(registry.entity_sets())} entity sets, "
            f"{generation_tasks} generation-enabled tasks"
        )
        return
    if args.command == "analyze":
        from .plotting import (
            load_analysis_spec,
            plot_boxplot_report,
            plot_ellipse_report,
            plot_radar_report,
        )

        analysis = load_analysis_spec(registry.config_dir)
        if args.analysis_command == "ellipses":
            output = args.output or (
                PROJECT_ROOT / "plots" / "ellipses" / f"{args.entity_set}.pdf"
            )
            plot_ellipse_report(
                registry,
                analysis,
                args.processed_outputs_dir,
                args.entity_set,
                output,
                task_ids=args.tasks,
                dimensions=args.dimensions,
                category_column=args.category_column,
                category_order=args.category_order,
                model_names=args.models,
                confidence=args.confidence,
                center=args.center,
                dpi=args.dpi,
            )
            return
        if args.analysis_command == "radar":
            entity_sets = args.entity_sets or list(analysis.entity_sets)
            output = args.output or PROJECT_ROOT / "plots" / "radar.pdf"
            plot_radar_report(
                registry,
                analysis,
                args.processed_outputs_dir,
                entity_sets,
                output,
                task_ids=args.tasks,
                model_names=args.models,
                dpi=args.dpi,
            )
            return
        output = args.output or (
            PROJECT_ROOT / "plots" / "boxplots" / f"{args.entity_set}_{args.metric}.pdf"
        )
        plot_boxplot_report(
            registry,
            analysis,
            args.processed_outputs_dir,
            args.entity_set,
            output,
            metric=args.metric,
            task_ids=args.tasks,
            category_column=args.category_column,
            category_order=args.category_order,
            model_names=args.models,
            center=args.center,
            dpi=args.dpi,
        )
        return
    if args.command == "server-info":
        for model in list_served_models(args.host, args.port, args.timeout):
            model_id = str(model["id"])
            print(f"Model ID:   {model_id}")
            print(f"Model name: {model_output_name(model_id)}")
            if model.get("root"):
                print(f"Model root: {model['root']}")
            if model.get("max_model_len"):
                print(f"Max length: {model['max_model_len']}")
        return
    if args.command == "prompt":
        task = registry.task(args.task)
        print(build_prompt(task, args.sentence, args.language, args.few_shot, args.numerical, args.target))
        return
    if args.command == "expand":
        task = registry.task(args.task)
        frame = expand_task(task, registry.entity_set(task.entity_set), args.language, args.variant)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame):,} rows to {args.output}")
        return
    if args.command == "generation-prompt":
        from .generation import build_generation_jobs, load_keywords, render_generation_prompt

        task = registry.task(args.task)
        if task.generation is None:
            raise ConfigError(f"Task '{task.id}' has no generation settings")
        label = task.label(args.label) if args.label else task.labels[0]
        keyword_pool = load_keywords(task.generation)
        jobs = build_generation_jobs(task, keyword_pool, seed=args.seed, samples_per_label=1)
        job = next(job for job in jobs if job.label_id == label.id)
        print(render_generation_prompt(task, job))
        return
    if args.command == "generate":
        from .generation import TextGenerationClient, generate_dataset

        task = registry.task(args.task)
        if task.generation is None:
            raise ConfigError(f"Task '{task.id}' has no generation settings")
        output = (args.output or task.datasets["main"]).resolve()
        client = TextGenerationClient(
            model=args.model,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            max_retries=args.request_retries,
            max_workers=args.workers,
        )
        model_name = model_output_name(client.model)
        print(f"Using server model: {client.model}")
        checkpoint = args.checkpoint or (
            PROJECT_ROOT
            / "outputs"
            / "generation"
            / task.id
            / model_name
            / "checkpoint-v2.csv"
        )
        generate_dataset(
            task,
            client,
            output,
            checkpoint,
            languages=args.languages,
            samples_per_label=args.samples_per_label,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_retries=args.validation_retries,
            overwrite=args.overwrite,
            allow_partial=args.allow_partial,
        )
        return


if __name__ == "__main__":
    main()
