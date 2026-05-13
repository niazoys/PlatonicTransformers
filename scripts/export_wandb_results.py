#!/usr/bin/env python
"""Export W&B run metrics to CSV, Markdown, and LaTeX.

Example:
  python scripts/export_wandb_results.py \
    --entity platonic-transformers-public \
    --project Platonic-ScanObjectNN-CamReady \
    --group-by model.solid_name model.attention \
    --metric-prefix test \
    --out-dir results/scanobjectnn
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import wandb


BASE_FIELDS = ["run_id", "run_name", "state", "sweep_id", "url"]


def nested_get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    return config.get(path, nested_get(config, path, default))


def parse_key_values(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_cli_config_args(args: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for arg in args:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, value = arg[2:].split("=", 1)
        parsed[key] = parse_value(value)
    return parsed


def local_run_config(run_id: str, roots: list[str]) -> dict[str, Any]:
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for metadata_path in root_path.glob(f"run-*-{run_id}/files/wandb-metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            config = parse_cli_config_args(metadata.get("args", []))
            if config:
                return config
    return {}


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Attention" if value else "Conv"
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return "Attention" if value.lower() == "true" else "Conv"
    return "" if value is None else str(value)


def metric_value(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_matches(name: str, prefixes: list[str], contains: list[str], regexes: list[re.Pattern[str]]) -> bool:
    if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
        return False
    if contains and not any(token in name for token in contains):
        return False
    if regexes and not any(regex.search(name) for regex in regexes):
        return False
    return True


def discover_metrics(rows: list[dict[str, Any]], requested: list[str]) -> list[str]:
    if requested:
        return requested
    metrics = sorted({key for row in rows for key in row if key not in BASE_FIELDS and not key.startswith("config.")})
    return metrics


def fmt_metric(values: list[float], percent: bool = True, latex: bool = False) -> str:
    if not values:
        return "--"
    scale = 100.0 if percent else 1.0
    scaled = [scale * value for value in values]
    if len(scaled) == 1:
        return f"{scaled[0]:.2f}"
    sep = r" \pm " if latex else " +/- "
    body = f"{mean(scaled):.2f}{sep}{stdev(scaled):.2f}"
    return f"${body}$" if latex else body


def build_filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if args.sweep_id:
        filters["sweep"] = args.sweep_id
    if args.tags:
        filters["tags"] = {"$all": args.tags}
    config_filters = parse_key_values(args.config_filter)
    for key, value in config_filters.items():
        filters[f"config.{key}"] = parse_value(value)
    return filters


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(f"{args.entity}/{args.project}", filters=filters)

    regexes = [re.compile(pattern) for pattern in args.metric_regex]
    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()

    for run in runs:
        if args.run_name_contains and args.run_name_contains not in run.name:
            continue
        if args.group and run.group != args.group:
            continue

        config = dict(run.config)
        config.update({key: value for key, value in local_run_config(run.id, args.local_wandb_dir).items() if config_get(config, key) is None})
        summary = dict(run.summary)
        row: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "sweep_id": run.sweep.id if run.sweep else "",
            "url": run.url,
        }

        for key in args.include_config + args.group_by:
            row[f"config.{key}"] = config_get(config, key)

        for key in args.metrics:
            row[key] = metric_value(summary, key)
            metric_names.add(key)

        if not args.metrics:
            for key in summary:
                if key.startswith("_"):
                    continue
                if metric_matches(key, args.metric_prefix, args.metric_contains, regexes):
                    value = metric_value(summary, key)
                    if value is not None:
                        row[key] = value
                        metric_names.add(key)

        rows.append(row)

    sort_keys = [f"config.{key}" for key in args.group_by] + [f"config.{key}" for key in args.include_config] + ["run_name"]
    rows.sort(key=lambda row: tuple(display_value(row.get(key)) for key in sort_keys))
    return rows, sorted(metric_names)


def write_raw_csv(rows: list[dict[str, Any]], metrics: list[str], args: argparse.Namespace, path: Path) -> None:
    config_fields = [f"config.{key}" for key in args.group_by + args.include_config]
    fields = BASE_FIELDS + list(dict.fromkeys(config_fields)) + metrics
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_values(
    rows: list[dict[str, Any]],
    group_by: list[str],
    metrics: list[str],
) -> dict[tuple[str, ...], dict[str, list[float]]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(display_value(row.get(f"config.{name}")) for name in group_by)
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                grouped[key][metric].append(float(value))
    return grouped


def metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def write_markdown_summary(rows: list[dict[str, Any]], metrics: list[str], args: argparse.Namespace, path: Path) -> None:
    if not args.group_by:
        path.write_text("No --group-by fields were provided.\n")
        return

    grouped = grouped_values(rows, args.group_by, metrics)
    header = [name.replace(".", " ") for name in args.group_by] + ["Runs"] + [metric_label(metric) for metric in metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key in sorted(grouped):
        values = grouped[key]
        run_count = max((len(values.get(metric, [])) for metric in metrics), default=0)
        line = list(key) + [str(run_count)] + [fmt_metric(values.get(metric, []), args.percent, latex=False) for metric in metrics]
        lines.append("| " + " | ".join(line) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex_summary(rows: list[dict[str, Any]], metrics: list[str], args: argparse.Namespace, path: Path) -> None:
    if not args.group_by:
        path.write_text("% No --group-by fields were provided.\n")
        return

    grouped = grouped_values(rows, args.group_by, metrics)
    cols = "l" * len(args.group_by) + "r" + "r" * len(metrics)
    header = [name.replace(".", " ") for name in args.group_by] + ["Runs"] + [metric_label(metric) for metric in metrics]
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    for key in sorted(grouped):
        values = grouped[key]
        run_count = max((len(values.get(metric, [])) for metric in metrics), default=0)
        line = list(key) + [str(run_count)] + [fmt_metric(values.get(metric, []), args.percent, latex=True) for metric in metrics]
        lines.append(" & ".join(line) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--group", default=None, help="Optional W&B run group filter.")
    parser.add_argument("--tags", nargs="*", default=[], help="Require all listed W&B tags.")
    parser.add_argument("--run-name-contains", default=None)
    parser.add_argument("--config-filter", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--group-by", nargs="*", default=[])
    parser.add_argument("--include-config", nargs="*", default=["seed"])
    parser.add_argument("--metrics", nargs="*", default=[], help="Exact metric names. Overrides discovery.")
    parser.add_argument("--metric-prefix", nargs="*", default=["test"])
    parser.add_argument("--metric-contains", nargs="*", default=[])
    parser.add_argument("--metric-regex", nargs="*", default=[])
    parser.add_argument("--out-dir", default="results/wandb_export")
    parser.add_argument("--basename", default="wandb_results")
    parser.add_argument("--local-wandb-dir", nargs="*", default=["mains/logs/wandb"], help="Local W&B directories used to recover CLI config args.")
    parser.add_argument("--no-percent", action="store_false", dest="percent", help="Do not multiply metric values by 100.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, discovered_metrics = collect_rows(args)
    metrics = discover_metrics(rows, args.metrics or discovered_metrics)
    write_raw_csv(rows, metrics, args, out_dir / f"{args.basename}_runs.csv")
    write_markdown_summary(rows, metrics, args, out_dir / f"{args.basename}_summary.md")
    write_latex_summary(rows, metrics, args, out_dir / f"{args.basename}_summary.tex")

    print(f"Exported {len(rows)} runs and {len(metrics)} metrics to {out_dir}")
    if metrics:
        print("Metrics:", ", ".join(metrics))


if __name__ == "__main__":
    main()
