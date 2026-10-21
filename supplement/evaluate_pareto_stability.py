"""
Leave-one-seed-out stability of the four empirical Pareto fronts.

Author: Gwenolé Quellec
Year: 2026

This module reproduces the four pairwise Pareto fronts from the original
constraint sweep and evaluates how their membership changes when one of the
five model seeds is excluded from the configuration-wise mean.

Analysis principles
-------------------
- use the exact five validation metrics and objective directions from
  ``scripts.constraint_sweep``;
- use all 200 configurations, not just the labeled representatives;
- reproduce the original Pareto dominance tolerance from ``search_protocol``;
- check the five-seed averages against saved aggregate sweep results;
- compare the five leave-one-seed-out fronts with the original five-seed front;
- report membership stability without claiming statistical confidence bounds;
- produce only tables, not an additional figure.

The analysis normally reads ``sweep_run_results.csv`` from the saved analysis
folder. ``--source runs`` instead reads each run's ``evaluation_metrics.json``
from the training folder. In both cases, the original analysis folder is used
for protocol and aggregate-result consistency checks.

Run from the repository root::

    python -m scripts.evaluate_pareto_stability \\
        --runs-dir runs-weight-sweep-final \\
        --analysis-dir analysis-weight-sweep-final \\
        --output-dir analysis-pareto-stability

No training, GPU, or additional model evaluation is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RUNS_DIR = Path("runs-weight-sweep-final")
DEFAULT_ANALYSIS_DIR = Path("analysis-weight-sweep-final")
DEFAULT_OUTPUT_DIR = Path("analysis-pareto-stability")
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_NUM_CONFIGURATIONS = 200

# Same names and directions as scripts.constraint_sweep. True means maximize.
OBJECTIVES = {
    "rollout_observation_mse_h5": False,
    "state_probe_r2": True,
    "neighborhood_trustworthiness": True,
    "counterfactual_relative_energy": False,
    "nuisance_latent_strong_class_balanced_accuracy": False,
}

# Same panel order as Figure 4 in the manuscript.
OBJECTIVE_PAIRS = (
    (
        "rollout_observation_mse_h5",
        "counterfactual_relative_energy",
    ),
    (
        "rollout_observation_mse_h5",
        "nuisance_latent_strong_class_balanced_accuracy",
    ),
    (
        "state_probe_r2",
        "counterfactual_relative_energy",
    ),
    (
        "state_probe_r2",
        "neighborhood_trustworthiness",
    ),
)

FRONT_LABELS = (
    "Prediction / counterfactual consistency",
    "Prediction / nuisance leakage",
    "State accessibility / counterfactual consistency",
    "State accessibility / neighborhood preservation",
)

METRIC_NAMES = tuple(OBJECTIVES)
WEIGHT_NAMES = (
    "predictive",
    "minimality",
    "temporal",
    "observation",
    "invariance",
    "structural",
)

# These tolerances apply to consistency checks, not to Pareto dominance.
AGGREGATE_RTOL = 1e-8
AGGREGATE_ATOL = 1e-10


# =============================================================================
# Input loading
# =============================================================================

def load_json(path: str | Path) -> object:
    """Read a JSON artifact without modifying the source file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Read flat rows from one existing sweep CSV."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Serialize flat records using stable, first-record column ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")

    fieldnames = list(rows[0])

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def parse_seed_manifest(
    analysis_dir: Path,
) -> dict[tuple[int, int], dict[str, object]]:
    """Index the original run manifest by configuration ID and model seed."""
    payload = load_json(analysis_dir / "sweep_manifest.json")

    if not isinstance(payload, list):
        raise TypeError("sweep_manifest.json must contain a JSON list.")

    records = {}

    for record in payload:
        key = (
            int(record["configuration_id"]),
            int(record["model_seed"]),
        )

        if key in records:
            raise ValueError(f"Duplicate manifest entry: {key}")

        records[key] = record

    return records


def load_runs_from_analysis(
    analysis_dir: Path,
) -> list[dict[str, object]]:
    """Read the original saved per-seed validation results (no reruns)."""
    path = analysis_dir / "sweep_run_results.csv"
    return list(load_csv(path))


def load_runs_from_directories(
    runs_dir: Path,
    manifest: Mapping[tuple[int, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Recover per-seed validation metrics from the original run folders."""
    result = []

    for key, record in sorted(manifest.items()):
        path = (
            runs_dir
            / str(record["run_name"])
            / "evaluation"
            / "evaluation_metrics.json"
        )
        payload = load_json(path)

        if not isinstance(payload, dict) or "validation" not in payload:
            raise KeyError(f"Missing validation split: {path}")

        metrics = payload["validation"]

        if float(metrics.get("nuisance_metric_version", -1)) != 2.0:
            raise ValueError(f"Unexpected nuisance metric version: {path}")

        result.append({
            **record,
            **{name: metrics[name] for name in METRIC_NAMES},
        })

    return result


def load_pareto_protocol(analysis_dir: Path) -> tuple[float, float]:
    """Read the dominance tolerances originally used to generate Figure 4."""
    protocol = load_json(analysis_dir / "search_protocol.json")

    if protocol.get("selection_split") != "validation":
        raise ValueError("The original sweep did not use the validation split.")

    arguments = protocol["arguments"]
    absolute = float(arguments["pareto_absolute_tolerance"])
    relative = float(arguments["pareto_relative_tolerance"])

    if absolute < 0 or relative < 0:
        raise ValueError("Pareto tolerances must be non-negative.")

    return absolute, relative


# =============================================================================
# Per-seed data validation
# =============================================================================

def build_metric_cube(
    records: Sequence[Mapping[str, object]],
    *,
    manifest: Mapping[tuple[int, int], Mapping[str, object]],
    seeds: Sequence[int],
    expected_configurations: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a complete (configuration, seed, metric) validation array."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("Model seed identifiers must be unique.")

    keys = sorted({
        int(record["configuration_id"])
        for record in records
    })

    if expected_configurations is not None and len(keys) != expected_configurations:
        raise ValueError(
            f"Expected {expected_configurations} configurations, found {len(keys)}."
        )

    if not keys:
        raise ValueError("No sweep configurations were loaded.")

    config_ids = np.asarray(keys, dtype=int)
    config_index = {config_id: index for index, config_id in enumerate(keys)}
    seed_index = {int(seed): index for index, seed in enumerate(seeds)}

    values = np.full(
        (len(keys), len(seeds), len(METRIC_NAMES)),
        np.nan,
        dtype=float,
    )

    seen = set()

    for record in records:
        config_id = int(record["configuration_id"])
        seed = int(record["model_seed"])
        key = (config_id, seed)

        if seed not in seed_index:
            continue

        if key in seen:
            raise ValueError(f"Duplicate run result: {key}")

        if key not in manifest:
            raise KeyError(f"Run absent from original sweep manifest: {key}")

        seen.add(key)

        values[config_index[config_id], seed_index[seed]] = [
            float(record[name]) for name in METRIC_NAMES
        ]

    missing = np.argwhere(~np.isfinite(values).all(axis=2))

    if missing.size:
        examples = [
            (int(config_ids[i]), int(seeds[j]))
            for i, j in missing[:10]
        ]
        raise ValueError(
            f"Missing or non-finite per-seed metrics in "
            f"{len(missing)} runs; examples: {examples}"
        )

    return config_ids, values


def check_original_aggregation(
    config_ids: np.ndarray,
    full_means: np.ndarray,
    analysis_dir: Path,
) -> None:
    """Ensure the reconstructed five-seed means match the paper's sweep."""
    records = load_csv(analysis_dir / "sweep_configuration_results.csv")
    reference = {
        int(record["configuration_id"]): record
        for record in records
    }

    if set(reference) != set(config_ids.tolist()):
        raise ValueError(
            "The aggregate CSV and per-seed results contain different configurations."
        )

    for index, config_id in enumerate(config_ids):
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            expected = float(reference[int(config_id)][metric_name])
            actual = float(full_means[index, metric_index])

            if not np.isclose(
                actual,
                expected,
                rtol=AGGREGATE_RTOL,
                atol=AGGREGATE_ATOL,
            ):
                raise ValueError(
                    "Saved aggregation mismatch for "
                    f"configuration {config_id}, {metric_name}: "
                    f"computed={actual:.12g}, saved={expected:.12g}."
                )


def load_paper_configuration_labels(analysis_dir: Path) -> dict[int, str]:
    """Recover the stable P labels used for representatives in Table 4."""
    path = analysis_dir / "pareto_configurations.csv"

    if not path.is_file():
        return {}

    return {
        int(record["configuration_id"]): record["label"]
        for record in load_csv(path)
    }


# =============================================================================
# Pareto dominance and stability statistics
# =============================================================================

def pareto_mask(
    values: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> np.ndarray:
    """Replicate the non-dominance rule in scripts.constraint_sweep.

    The input contains minimized objectives. Dominance is evaluated relative
    to each candidate, using the original candidate-specific tolerance rule.
    This deliberately retains the published sweep's Pareto convention.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected two minimized objectives per configuration.")

    result = np.ones(len(values), dtype=bool)

    for index, candidate in enumerate(values):
        tolerance = (
            absolute_tolerance
            + relative_tolerance * np.maximum(np.abs(candidate), 1.0)
        )
        no_worse = np.all(values <= candidate + tolerance, axis=1)
        strictly_better = np.any(values < candidate - tolerance, axis=1)
        dominating = no_worse & strictly_better
        dominating[index] = False

        if np.any(dominating):
            result[index] = False

    return result


def oriented_pair(
    all_metrics: np.ndarray,
    x_metric: str,
    y_metric: str,
) -> np.ndarray:
    """Select paper metric columns and convert maximization to minimization."""
    metric_indices = (
        METRIC_NAMES.index(x_metric),
        METRIC_NAMES.index(y_metric),
    )
    signs = np.asarray([
        -1.0 if OBJECTIVES[name] else 1.0
        for name in (x_metric, y_metric)
    ])
    return all_metrics[:, metric_indices] * signs


def front_membership(
    metric_means: np.ndarray,
    x_metric: str,
    y_metric: str,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    """Identify the front across *all* configurations for one metric pair."""
    return pareto_mask(
        oriented_pair(metric_means, x_metric, y_metric),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def front_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    """Compute Jaccard similarity and reference-front retention."""
    intersection = int(np.count_nonzero(reference & candidate))
    union = int(np.count_nonzero(reference | candidate))
    n_reference = int(np.count_nonzero(reference))

    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
        "retention": intersection / n_reference if n_reference else 1.0,
        "n_front": int(np.count_nonzero(candidate)),
    }


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return descriptive mean and sample standard deviation."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def check_original_fronts(
    counts: Mapping[str, int],
    analysis_dir: Path,
) -> None:
    """Check original five-seed front sizes against the saved Figure 4 report."""
    path = analysis_dir / "pareto_2d_summary.json"

    if not path.is_file():
        print("Warning: pareto_2d_summary.json missing; skipping front-count check.")
        return

    records = load_json(path)

    if len(records) != len(OBJECTIVE_PAIRS):
        raise ValueError("Unexpected number of fronts in pareto_2d_summary.json")

    for panel_index, ((x_metric, y_metric), record) in enumerate(
        zip(OBJECTIVE_PAIRS, records, strict=True)
    ):
        panel = chr(97 + panel_index)

        if (record["x_metric"], record["y_metric"]) != (x_metric, y_metric):
            raise ValueError(f"Different metric pair for original panel ({panel}).")

        expected = int(record["n_pareto"])
        actual = counts[panel]

        if expected != actual:
            raise ValueError(
                f"Front ({panel}) mismatch: saved {expected}, recomputed {actual}. "
                "Check source metrics and Pareto tolerances."
            )


def analyze_stability(
    config_ids: np.ndarray,
    metrics: np.ndarray,
    *,
    seeds: Sequence[int],
    absolute_tolerance: float,
    relative_tolerance: float,
    labels: Mapping[int, str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, int],
]:
    """Compare each full five-seed front to five four-seed fronts."""
    full_means = metrics.mean(axis=1)
    loo_mean_arrays = [
        np.delete(metrics, seed_index, axis=1).mean(axis=1)
        for seed_index in range(len(seeds))
    ]

    summary_rows = []
    leave_one_out_rows = []
    membership_rows = []
    reference_counts = {}

    for pair_index, (x_metric, y_metric) in enumerate(OBJECTIVE_PAIRS):
        panel = chr(97 + pair_index)
        front_name = FRONT_LABELS[pair_index]

        reference = front_membership(
            full_means,
            x_metric,
            y_metric,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        n_reference = int(reference.sum())
        reference_counts[panel] = n_reference
        loo_masks = []
        similarities = []

        for seed, loo_means in zip(seeds, loo_mean_arrays, strict=True):
            candidate = front_membership(
                loo_means,
                x_metric,
                y_metric,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            loo_masks.append(candidate)
            comparison = front_similarity(reference, candidate)
            similarities.append(comparison)

            leave_one_out_rows.append({
                "panel": panel,
                "front": front_name,
                "omitted_seed": int(seed),
                "reference_front_size": n_reference,
                "loo_front_size": comparison["n_front"],
                "intersection_size": comparison["intersection"],
                "union_size": comparison["union"],
                "jaccard": comparison["jaccard"],
                "retained_reference_fraction": comparison["retention"],
            })

        jaccards = [float(s["jaccard"]) for s in similarities]
        retentions = [float(s["retention"]) for s in similarities]
        front_sizes = [float(s["n_front"]) for s in similarities]
        jaccard_mean, jaccard_sd = mean_sd(jaccards)
        retention_mean, retention_sd = mean_sd(retentions)
        size_mean, size_sd = mean_sd(front_sizes)

        # Each original front member can be retained once per seed omission.
        # These are repeated sensitivity checks, not independent experiments.
        retained_total = sum(int(s["intersection"]) for s in similarities)
        retention_opportunities = n_reference * len(seeds)

        summary_rows.append({
            "panel": panel,
            "front": front_name,
            "n_reference_front": n_reference,
            "n_leave_one_out": len(seeds),
            "retained_total": retained_total,
            "retention_opportunities": retention_opportunities,
            "n_loo_front_mean": size_mean,
            "n_loo_front_sd": size_sd,
            "jaccard_mean": jaccard_mean,
            "jaccard_sd": jaccard_sd,
            "jaccard_min": min(jaccards),
            "jaccard_max": max(jaccards),
            "reference_retained_pct_mean": 100 * retention_mean,
            "reference_retained_pct_sd": 100 * retention_sd,
            "reference_retained_pct_min": 100 * min(retentions),
            "reference_retained_pct_max": 100 * max(retentions),
        })

        membership_frequency = np.asarray(loo_masks, dtype=int).mean(axis=0)

        for index, config_id in enumerate(config_ids):
            membership_rows.append({
                "panel": panel,
                "configuration_id": int(config_id),
                "label": labels.get(int(config_id), ""),
                "on_five_seed_front": int(reference[index]),
                "loo_front_count": int(sum(mask[index] for mask in loo_masks)),
                "loo_front_frequency": float(membership_frequency[index]),
            })

    return summary_rows, leave_one_out_rows, membership_rows, reference_counts


# =============================================================================
# Supplementary table and optional per-configuration uncertainties
# =============================================================================

def format_three_decimals(value: float) -> str:
    """Round displayed descriptive statistics to three decimals (half up)."""
    return str(
        Decimal(str(value)).quantize(
            Decimal("0.001"),
            rounding=ROUND_HALF_UP,
        )
    )


def write_supplementary_table(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Write a four-front table with observed Jaccard ranges and retention counts.

    Each original front member contributes one possible retained membership per
    omitted seed. For example, nine retained memberships for two front members
    across five omissions are reported as 90.0% (9/10). These checks are
    descriptive and are not treated as independent Bernoulli trials.
    """
    tex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{\textbf{Leave-one-seed-out stability of empirical pairwise Pareto fronts.} "
        r"Each of the five analyses excludes one model seed. "
        r"Jaccard values are the mean and observed range (minimum--maximum) "
        r"of similarity to the original five-seed front. "
        r"Retention is the percentage and total number of original-front "
        r"memberships preserved across the five analyses, divided by "
        r"$5\times N_{\mathrm{front}}$. Memberships are counted repeatedly "
        r"across omissions and do not represent independent experiments.}",
        r"  \label{tab:pareto-stability}",
        r"  \begin{tabular}{lrrr}",
        r"    \hline",
        r"    Front & $N_{\mathrm{front}}$ & Jaccard: mean [min--max] & Retained \% ($n/N$) \\",
        r"    \hline",
    ]
    md_lines = [
        "| Front | N (five seeds) | Jaccard mean [min–max] | Retained (%) (n/N) |",
        "|:---|---:|---:|---:|",
    ]

    for row in rows:
        panel = str(row["panel"])
        count = int(row["n_reference_front"])
        jm = float(row["jaccard_mean"])
        jmin = float(row["jaccard_min"])
        jmax = float(row["jaccard_max"])
        rm = float(row["reference_retained_pct_mean"])
        jmean_text = format_three_decimals(jm)
        jmin_text = format_three_decimals(jmin)
        jmax_text = format_three_decimals(jmax)
        n_retained = int(row["retained_total"])
        n_opportunities = int(row["retention_opportunities"])

        if not np.isclose(
            rm,
            100.0 * n_retained / n_opportunities,
            rtol=1e-12,
            atol=1e-10,
        ):
            raise ValueError(f"Inconsistent retained membership count for ({panel}).")

        tex_lines.append(
            f"    ({panel}) & {count} & "
            f"${jmean_text}$ [{jmin_text}--{jmax_text}] & "
            f"{rm:.1f}\\% ({n_retained}/{n_opportunities}) " + r"\\"
        )
        md_lines.append(
            f"| ({panel}) | {count} | "
            f"{jmean_text} [{jmin_text}–{jmax_text}] | "
            f"{rm:.1f}% ({n_retained}/{n_opportunities}) |"
        )

    tex_lines.extend([
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    (output_dir / "pareto_stability_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "pareto_stability_table.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def representative_metric_uncertainty(
    config_ids: np.ndarray,
    metrics: np.ndarray,
    labels: Mapping[int, str],
) -> list[dict[str, object]]:
    """Save five-seed mean/SD for the configurations labeled in Table 4."""
    rows = []
    order = {int(cid): index for index, cid in enumerate(config_ids)}
    sorted_ids = sorted(
        (config_id for config_id in labels if config_id in order),
        key=lambda cid: int(labels[cid][1:]) if labels[cid].startswith("P") else cid,
    )

    for config_id in sorted_ids:
        values = metrics[order[config_id]]
        row = {
            "configuration_id": int(config_id),
            "label": labels[config_id],
            "n_model_seeds": int(values.shape[0]),
        }

        for index, name in enumerate(METRIC_NAMES):
            row[f"{name}_mean"] = float(values[:, index].mean())
            row[f"{name}_sd"] = float(values[:, index].std(ddof=1))

        rows.append(row)

    return rows


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Define input locations and repeatability checks."""
    parser = argparse.ArgumentParser(
        description="Leave-one-seed-out stability of the four paper Pareto fronts."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_MODEL_SEEDS,
    )
    parser.add_argument(
        "--num-configurations",
        type=int,
        default=DEFAULT_NUM_CONFIGURATIONS,
        help="Expected configuration count, to detect incomplete results.",
    )
    parser.add_argument(
        "--source",
        choices=("analysis", "runs"),
        default="analysis",
        help="Read per-seed metrics from the analysis CSV (default) or run JSONs.",
    )
    parser.add_argument(
        "--verify-run-files",
        action="store_true",
        help="Verify all evaluation JSON files exist in the training directory.",
    )
    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """Rebuild and audit the original fronts, then calculate LOO stability."""
    args = build_arg_parser().parse_args()
    print(f"Running script: {Path(__file__).resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(seed) for seed in args.model_seeds)
    manifest = parse_seed_manifest(args.analysis_dir)
    absolute_tolerance, relative_tolerance = load_pareto_protocol(args.analysis_dir)

    if args.source == "runs":
        print("Loading per-seed evaluation JSON files from training runs ...")
        records = load_runs_from_directories(args.runs_dir, manifest)
    else:
        print("Loading original per-seed sweep_run_results.csv ...")
        records = load_runs_from_analysis(args.analysis_dir)

    if args.verify_run_files:
        missing = [
            str(args.runs_dir / str(record["run_name"]) / "evaluation" / "evaluation_metrics.json")
            for key, record in manifest.items()
            if key[1] in seeds
            and not (args.runs_dir / str(record["run_name"]) / "evaluation" / "evaluation_metrics.json").is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} evaluation files, e.g. {missing[:3]}"
            )

    config_ids, metric_cube = build_metric_cube(
        records,
        manifest=manifest,
        seeds=seeds,
        expected_configurations=args.num_configurations,
    )

    check_original_aggregation(
        config_ids,
        metric_cube.mean(axis=1),
        args.analysis_dir,
    )

    labels = load_paper_configuration_labels(args.analysis_dir)
    summary, leave_one_out, membership, counts = analyze_stability(
        config_ids,
        metric_cube,
        seeds=seeds,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        labels=labels,
    )
    check_original_fronts(counts, args.analysis_dir)

    save_csv(summary, args.output_dir / "pareto_stability_summary.csv")
    save_csv(leave_one_out, args.output_dir / "pareto_stability_leave_one_out.csv")
    save_csv(membership, args.output_dir / "pareto_stability_membership.csv")

    uncertainties = representative_metric_uncertainty(
        config_ids, metric_cube, labels
    )

    if uncertainties:
        save_csv(
            uncertainties,
            args.output_dir / "representative_metric_uncertainty.csv",
        )

    write_supplementary_table(summary, args.output_dir)

    report = {
        "source": args.source,
        "runs_dir": str(args.runs_dir),
        "analysis_dir": str(args.analysis_dir),
        "n_configurations": int(len(config_ids)),
        "model_seeds": list(seeds),
        "selection_split": "validation",
        "pareto_absolute_tolerance": absolute_tolerance,
        "pareto_relative_tolerance": relative_tolerance,
        "n_leave_one_out_repetitions": len(seeds),
        "fronts": summary,
        "interpretation": (
            "Descriptive sensitivity analysis; not confidence intervals "
            "or probabilities of Pareto optimality."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Verified original five-seed fronts: {counts}")
    print("Leave-one-seed-out Pareto stability:")

    for row in summary:
        print(
            f"  ({row['panel']}) N={row['n_reference_front']:3d} | "
            f"Jaccard={row['jaccard_mean']:.3f} "
            f"+/- {row['jaccard_sd']:.3f} | "
            f"retained={row['reference_retained_pct_mean']:.1f}% "
            f"({row['retained_total']}/{row['retention_opportunities']})"
        )

    print(f"Results saved to {args.output_dir}/")
    print(f"Table: {args.output_dir / 'pareto_stability_table.tex'}")


if __name__ == "__main__":
    main()
