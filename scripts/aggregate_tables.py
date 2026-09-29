"""
Build paper and exhaustive tables from aggregate CLSM evaluations.

Author: Gwenolé Quellec
Year: 2026

The script reads every ``*-aggregate-evaluation.json`` file from one
directory and writes:

- a compact LaTeX table for the manuscript;
- an exhaustive Markdown table containing all aggregate metrics.

Examples
--------
Generate both tables from the final preset evaluation directory:

    python -m scripts.aggregate_tables runs

This writes:

    runs/aggregate-table.tex
    runs/aggregate-table.md

Custom output paths can also be provided:

    python -m scripts.aggregate_tables runs \
        --latex-output tables/main-results.tex \
        --markdown-output tables/all-results.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


# =============================================================================
# Paper table specification
# =============================================================================

PAPER_METRICS = (
    (
        "test",
        "rollout_observation_mse_h5",
        r"\makecell{Pred. \\ MSE \\ $h=5 \downarrow$}",
        "min",
    ),
    (
        "test",
        "state_probe_r2",
        r"\makecell{State \\ $R^2 \uparrow$}",
        "max",
    ),
    (
        "test",
        "neighborhood_trustworthiness",
        r"\makecell{Trust. \\ $\uparrow$}",
        "max",
    ),
    (
        "test",
        "counterfactual_relative_energy",
        r"\makecell{CF \\ energy $\downarrow$}",
        "min",
    ),
    (
        "test",
        "nuisance_latent_strong_class_balanced_accuracy",
        r"\makecell{Nuis. \\ $Z$ \\ BA $\downarrow$}",
        "min",
    ),
    (
        "test",
        "nuisance_joint_strong_class_balanced_accuracy",
        r"\makecell{Nuis. \\ $(S,Z)$ \\ BA $\downarrow$}",
        "min",
    ),
    (
        "ood",
        "rollout_observation_mse_h5",
        r"\makecell{OOD \\ pred. MSE \\ $h=5 \downarrow$}",
        "min",
    ),
    (
        "ood",
        "state_probe_r2",
        r"\makecell{OOD \\ state \\ $R^2 \uparrow$}",
        "max",
    ),
)


# =============================================================================
# Input utilities
# =============================================================================

def configuration_name(
    path: Path,
) -> str:
    """Extract the configuration name from an aggregate-evaluation filename."""

    suffix = "-aggregate-evaluation.json"

    if not path.name.endswith(
        suffix
    ):
        raise ValueError(
            f"Unexpected aggregate filename: {path.name}"
        )

    return path.name[
        : -len(suffix)
    ]


def natural_configuration_key(
    name: str,
):
    """
    Sort P1, P2, ..., P10 numerically while keeping named controls readable.
    """

    match = re.fullmatch(
        r"P(\d+)",
        name,
    )

    if match is not None:
        return (
            1,
            int(
                match.group(1)
            ),
            name,
        )

    control_order = {
        "predictive": 0,
        "reconstruction": 1,
        "temporal": 2,
        "structural": 3,
    }

    if name in control_order:
        return (
            0,
            control_order[name],
            name,
        )

    return (
        2,
        name,
    )


def load_aggregate(
    path: Path,
) -> dict:
    """Load and validate one aggregate evaluation file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        payload = json.load(
            stream
        )

    aggregate = payload.get(
        "aggregate"
    )

    if not isinstance(
        aggregate,
        dict,
    ):
        raise ValueError(
            f"{path} does not contain an 'aggregate' object."
        )

    return aggregate


def load_directory(
    directory: Path,
) -> dict[str, dict]:
    """Load every aggregate evaluation JSON found in a directory."""

    paths = sorted(
        directory.glob(
            "*-aggregate-evaluation.json"
        ),
        key=lambda path: natural_configuration_key(
            configuration_name(
                path
            )
        ),
    )

    if not paths:
        raise FileNotFoundError(
            "No *-aggregate-evaluation.json files found in "
            f"{directory}"
        )

    evaluations = {}

    for path in paths:
        name = configuration_name(
            path
        )

        if name in evaluations:
            raise ValueError(
                f"Duplicate configuration name: {name}"
            )

        evaluations[name] = load_aggregate(
            path
        )

    return evaluations


# =============================================================================
# Numeric formatting
# =============================================================================

def format_significant(
    value,
    digits: int = 3,
) -> str:
    """Format one finite number with a fixed number of significant digits."""

    if value is None:
        return "--"

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        return "--"

    if value == 0.0:
        return "0"

    return format(
        value,
        f".{digits}g",
    )


def format_mean(
    statistics: dict | None,
    digits: int = 3,
) -> str:
    """Format an aggregate mean with significant digits."""

    if not statistics:
        return "--"

    return format_significant(
        statistics.get("mean"),
        digits,
    )


def markdown_value(
    value,
) -> str:
    """Preserve JSON numeric precision in the exhaustive Markdown table."""

    if value is None:
        return "—"

    if isinstance(
        value,
        float,
    ) and not math.isfinite(
        value
    ):
        return "—"

    return str(
        value
    )


# =============================================================================
# LaTeX paper table
# =============================================================================

def paper_statistic(
    aggregate: dict,
    split: str,
    metric: str,
) -> dict | None:
    """Retrieve one aggregate statistic block."""

    split_metrics = aggregate.get(
        split
    )

    if not isinstance(
        split_metrics,
        dict,
    ):
        return None

    statistics = split_metrics.get(
        metric
    )

    if not isinstance(
        statistics,
        dict,
    ):
        return None

    return statistics


def build_latex_table(
    evaluations: dict[str, dict],
) -> str:
    """Build the compact manuscript table."""

    column_count = (
        1
        + len(PAPER_METRICS)
    )

    best_values = {}

    for (
        split,
        metric,
        _,
        direction,
    ) in PAPER_METRICS:
        values = []

        for aggregate in evaluations.values():
            statistics = paper_statistic(
                aggregate,
                split,
                metric,
            )

            if (
                statistics is not None
                and statistics.get("mean") is not None
            ):
                values.append(
                    float(
                        statistics["mean"]
                    )
                )

        if not values:
            best_values[
                (
                    split,
                    metric,
                )
            ] = None

        elif direction == "min":
            best_values[
                (
                    split,
                    metric,
                )
            ] = min(
                values
            )

        elif direction == "max":
            best_values[
                (
                    split,
                    metric,
                )
            ] = max(
                values
            )

        else:
            raise ValueError(
                f"Unknown optimization direction: {direction}"
            )

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        (
            r"  \begin{tabular}{l"
            + "c" * (
                column_count - 1
            )
            + "}"
        ),
        r"    \toprule",
    ]

    header = (
        ["Configuration"]
        + [
            label
            for _, _, label, _ in PAPER_METRICS
        ]
    )

    lines.append(
        "    "
        + " & ".join(
            header
        )
        + r" \\"
    )

    lines.append(
        r"    \midrule"
    )

    for name, aggregate in evaluations.items():
        row = [
            name
        ]

        for (
            split,
            metric,
            _,
            _,
        ) in PAPER_METRICS:
            statistics = paper_statistic(
                aggregate,
                split,
                metric,
            )

            value = format_mean(
                statistics,
                digits=3,
            )

            mean = (
                None
                if statistics is None
                else statistics.get(
                    "mean"
                )
            )

            best = best_values[
                (
                    split,
                    metric,
                )
            ]

            if (
                mean is not None
                and best is not None
                and float(mean) == best
            ):
                value = (
                    rf"\textbf{{{value}}}"
                )

            row.append(
                value
            )

        lines.append(
            "    "
            + " & ".join(
                row
            )
            + r" \\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{",
            r"    \textbf{Performance of the selected CLSM configurations.}",
            r"    Values are means across model seeds. Best values in each column are shown in bold.",
            r"  }",
            r"  \label{tab:clsm-main-results}",
            r"\end{table*}",
            "",
        ]
    )

    return "\n".join(
        lines
    )


# =============================================================================
# Exhaustive Markdown table
# =============================================================================

def build_markdown_table(
    evaluations: dict[str, dict],
) -> str:
    """
    Build a long-format table containing every aggregate metric.

    Long format avoids an impractically wide table as the evaluation protocol
    grows.
    """

    lines = [
        "# Aggregate CLSM evaluation",
        "",
        (
            "| Configuration | Split | Metric | "
            "n | Mean | Std | Min | Max |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|"
        ),
    ]

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
        "ood": 3,
    }

    for name, aggregate in evaluations.items():
        splits = sorted(
            aggregate,
            key=lambda split: (
                split_order.get(
                    split,
                    99,
                ),
                split,
            ),
        )

        for split in splits:
            metrics = aggregate[
                split
            ]

            if not isinstance(
                metrics,
                dict,
            ):
                continue

            for metric in sorted(
                metrics
            ):
                statistics = metrics[
                    metric
                ]

                if not isinstance(
                    statistics,
                    dict,
                ):
                    continue

                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            split,
                            metric,
                            markdown_value(
                                statistics.get(
                                    "n"
                                )
                            ),
                            markdown_value(
                                statistics.get(
                                    "mean"
                                )
                            ),
                            markdown_value(
                                statistics.get(
                                    "std"
                                )
                            ),
                            markdown_value(
                                statistics.get(
                                    "min"
                                )
                            ),
                            markdown_value(
                                statistics.get(
                                    "max"
                                )
                            ),
                        ]
                    )
                    + " |"
                )

    lines.append(
        ""
    )

    return "\n".join(
        lines
    )


# =============================================================================
# Command-line interface
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "directory",
        type=Path,
        help=(
            "Directory containing "
            "*-aggregate-evaluation.json files."
        ),
    )

    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=(
            "LaTeX output path. Defaults to "
            "<directory>/aggregate-table.tex."
        ),
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=(
            "Markdown output path. Defaults to "
            "<directory>/aggregate-table.md."
        ),
    )

    return parser


def main() -> None:
    """Generate the paper and exhaustive aggregate tables."""

    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(
            f"Not a directory: {args.directory}"
        )

    latex_output = (
        args.latex_output
        if args.latex_output is not None
        else args.directory
        / "aggregate-table.tex"
    )

    markdown_output = (
        args.markdown_output
        if args.markdown_output is not None
        else args.directory
        / "aggregate-table.md"
    )

    evaluations = load_directory(
        args.directory
    )

    latex_output.write_text(
        build_latex_table(
            evaluations
        ),
        encoding="utf-8",
    )

    markdown_output.write_text(
        build_markdown_table(
            evaluations
        ),
        encoding="utf-8",
    )

    print(
        f"Configurations : {len(evaluations)}"
    )
    print(
        f"LaTeX          : {latex_output}"
    )
    print(
        f"Markdown       : {markdown_output}"
    )


if __name__ == "__main__":
    main()

