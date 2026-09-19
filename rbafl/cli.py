"""Command-line entry point for the public RB-AFL reproduction workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .study import (
    EXPECTED_TRAINING_SEEDS,
    aggregate_ablation_stage,
    benchmark_efficiency_stage,
    cache_attacks_stage,
    calibrate_stage,
    config_paths,
    evaluate_stage,
    figures_stage,
    load_config,
    prepare_stage,
    run_manifest,
    split_stage,
    train_stage,
    uniqueness_stage,
    validate_split_file,
    write_protocol_lock,
)


def _parse_seeds(value: str) -> list[int] | None:
    text = value.strip()
    if not text:
        return None
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    if not set(seeds).issubset(EXPECTED_TRAINING_SEEDS):
        raise ValueError("Seeds must be selected from 20260730..20260739")
    return seeds


def _parse_experiments(value: str) -> list[str] | None:
    text = value.strip()
    return [item.strip().upper() for item in text.split(",") if item.strip()] or None


def _print_result(result: object) -> None:
    if isinstance(result, Path):
        print(result)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        for item in result:
            print(item)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rbafl",
        description="RB-AFL reproducibility pipeline: 50 train / 20 calibration / 41 test",
    )
    parser.add_argument(
        "command",
        choices=(
            "validate-config",
            "prepare",
            "split",
            "validate-split",
            "train-e5",
            "train-all",
            "calibrate",
            "cache-attacks",
            "evaluate",
            "uniqueness",
            "efficiency",
            "aggregate",
            "figures",
            "manifest",
            "all",
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/protocol_50_20_41.json",
        help="Path to the frozen JSON protocol configuration",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated subset of the ten frozen training seeds",
    )
    parser.add_argument(
        "--experiments",
        default="",
        help="Optional comma-separated ablations, e.g. E1,E4,E5",
    )
    parser.add_argument(
        "--replace-split",
        action="store_true",
        help="Regenerate the split using the same frozen protocol (never use after viewing test results)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seeds = _parse_seeds(args.seeds)
    experiments = _parse_experiments(args.experiments)
    command = args.command

    if command == "validate-config":
        paths = {name: str(value) if value is not None else None for name, value in config_paths(config).items()}
        _print_result({"status": "ok", "config_sha256": config["_config_sha256"], "paths": paths})
        return
    if command == "prepare":
        _print_result(prepare_stage(config))
        return
    if command == "split":
        _print_result(split_stage(config, reuse_existing=not args.replace_split))
        return
    if command == "validate-split":
        _print_result(validate_split_file(config_paths(config)["split"]))
        return
    if command == "train-e5":
        _print_result(train_stage(config, seeds=seeds, experiments=["E5"]))
        return
    if command == "train-all":
        _print_result(train_stage(config, seeds=seeds, experiments=experiments))
        return
    if command == "calibrate":
        _print_result(calibrate_stage(config))
        return
    if command == "cache-attacks":
        _print_result(cache_attacks_stage(config))
        return
    if command == "evaluate":
        _print_result(evaluate_stage(config, seeds=seeds))
        return
    if command == "uniqueness":
        _print_result(uniqueness_stage(config))
        return
    if command == "efficiency":
        _print_result(benchmark_efficiency_stage(config, seeds=seeds))
        return
    if command == "aggregate":
        _print_result(aggregate_ablation_stage(config))
        return
    if command == "figures":
        _print_result(figures_stage(config))
        return
    if command == "manifest":
        _print_result(run_manifest(config))
        return

    # Full order is deliberately leakage-safe: the threshold is frozen before
    # any final-test attack, uniqueness, ablation, or efficiency stage runs.
    prepare_stage(config)
    split_stage(config, reuse_existing=True)
    train_stage(config, experiments=["E5"])
    calibrate_stage(config)
    train_stage(config, experiments=["E1", "E2", "E3", "E4", "E6", "E7"])
    cache_attacks_stage(config)
    evaluate_stage(config)
    uniqueness_stage(config)
    benchmark_efficiency_stage(config)
    aggregate_ablation_stage(config)
    figures_stage(config)
    write_protocol_lock(config)
    _print_result(run_manifest(config))


if __name__ == "__main__":
    main()
