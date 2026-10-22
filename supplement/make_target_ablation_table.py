"""Generate a reproducible target-selection/invariance ablation table.

Compare the original full-target configuration with its reduced-target
variants, with and without adversarial invariance. All displayed statistics are
recalculated from five individual training seeds.

Scope
-----
This script implements the target-selection ablation protocol of the CLSM-Toy
benchmark (12 input channels, 6 prediction targets). It supports different
configurations within this benchmark but is not intended as a general-purpose
ablation tool for other datasets.

Example
-------

    python -m scripts.make_target_ablation_table \\
        --configuration CONFIG_ID \\
        --input-dir runs-observable-target6 \\
        --output-dir runs-observable-target6 \\
        --require-original-evaluation

No training or probe fitting is performed. A missing original evaluation may
be tolerated for a clearly marked draft (omit --require-original-evaluation).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

SEEDS = set(range(5))
CHANNELS6 = list(range(2, 8))
VARIANTS = ("original", "target6", "target6_noinv")

# Section, row heading (Markdown), LaTeX label, JSON key, source, split,
# multiplier, number of decimals.
SPECS = [
    ("Representation and invariance", "Nuisance BA from Z (%)", "Nuisance BA from $Z$ (\\%)", "nuisance_latent_strong_class_balanced_accuracy", "evaluation", "test", 100, 2),
    ("Representation and invariance", "Nuisance BA from (S,Z) (%)", "Nuisance BA from $(S,Z)$ (\\%)", "nuisance_joint_strong_class_balanced_accuracy", "evaluation", "test", 100, 2),
    ("Representation and invariance", "Conditional BA gain over S (pp)", "BA gain over $S$ (percentage points)", "nuisance_joint_over_state_strong_balanced_accuracy_gain", "evaluation", "test", 100, 2),
    ("Representation and invariance", "Continuous nuisance R² from (S,Z)", "Continuous nuisance $R^2$ from $(S,Z)$", "nuisance_joint_nonlinear_continuous_r2", "evaluation", "test", 1, 4),
    ("Representation and invariance", "Linear physical-state R²", "Linear physical-state $R^2$", "state_probe_r2", "evaluation", "test", 1, 3),
    ("Representation and invariance", "Nonlinear future-state R² (h=5)", "Nonlinear future-state $R^2$ ($h=5$)", "physical_nonlinear_r2_h5", "evaluation", "test", 1, 3),
    ("Representation and invariance", "Counterfactual relative energy (ID)", "Counterfactual relative energy (ID)", "counterfactual_relative_energy", "evaluation", "test", 1, 4),
    ("Prediction on six common channels", "Six-channel rollout MSE (h=5, ID)", "Six-channel rollout MSE ($h=5$, ID)", "rollout_observation_target6_mse_h5", "mse", "test", 1, 5),
    ("Prediction on six common channels", "Six-channel rollout MSE (h=5, OOD)", "Six-channel rollout MSE ($h=5$, OOD)", "rollout_observation_target6_mse_h5", "mse", "ood", 1, 5),
]


# =============================================================================
# Data loading
# =============================================================================

def parse_configuration(value: str) -> str:
    """Accept a safe single-token experiment name for file and LaTeX labels."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "configuration must start with a letter and contain only letters, digits, '_' or '-'"
        )
    return value


def file_names(configuration: str) -> dict[str, dict[str, str]]:
    return {
        "original": {
            "evaluation": f"{configuration}-original-evaluation.json",
            "mse": f"{configuration}-original_mse6.json",
        },
        "target6": {
            "evaluation": f"{configuration}-target6-evaluation.json",
            "mse": f"{configuration}-target6_mse6.json",
        },
        "target6_noinv": {
            "evaluation": f"{configuration}-target6-noinv-evaluation.json",
            "mse": f"{configuration}-target6-noinv_mse6.json",
        },
    }


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def get_per_seed(raw: dict, source: str, split: str, metric: str) -> dict[int, float]:
    records = raw["runs"] if source == "evaluation" else raw["by_seed"]
    vals: dict[int, float] = {}
    for name, record in records.items():
        if source == "evaluation":
            match = re.search(r"-seed-(\d+)$", name)
            if match is None:
                raise ValueError(f"Unrecognized run ID: {name}")
            seed = int(match.group(1))
        else:
            seed = int(name)
        if seed in vals:
            raise ValueError(f"Duplicate seed {seed}")
        v = record[split][metric]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Missing/nonfinite value for seed {seed}, {split}/{metric}")
        vals[seed] = float(v)
    if set(vals) != SEEDS:
        raise ValueError(f"Expected seeds {sorted(SEEDS)}, got {sorted(vals)}")
    return vals


def format_value(vals: dict[int, float] | None, scale: int, ndigits: int, tex: bool) -> str:
    if vals is None:
        return r"\textemdash{}" if tex else "—"
    numbers = [vals[i] * scale for i in sorted(SEEDS)]
    center = statistics.mean(numbers)
    sd = statistics.stdev(numbers)
    if abs(center) < 0.5 * 10 ** -ndigits:
        center = 0.0
    core = f"{center:.{ndigits}f} \\pm {sd:.{ndigits}f}" if tex else f"{center:.{ndigits}f} ± {sd:.{ndigits}f}"
    return f"${core}$" if tex else core


def check_inputs(data: dict[str, dict[str, dict | None]], configuration: str,
                 require_original: bool) -> bool:
    orig_available = data["original"]["evaluation"] is not None
    if require_original and not orig_available:
        raise ValueError(
            f"{configuration}-original-evaluation.json is required for a complete publication table"
        )
    hashes = []
    for variant, group in data.items():
        mse = group["mse"]
        assert mse is not None
        if mse.get("configuration") != configuration:
            raise ValueError(f"Configuration mismatch in MSE for {variant}")
        if mse.get("target_channels_zero_based") != CHANNELS6:
            raise ValueError(f"Unexpected evaluated channels for {variant}")
        if group["evaluation"] is not None:
            get_per_seed(group["evaluation"], "evaluation", "test", "state_probe_r2")
        get_per_seed(mse, "mse", "test", "rollout_observation_target6_mse_h5")
        get_per_seed(mse, "mse", "ood", "rollout_observation_target6_mse_h5")
        hashes.append(mse.get("dataset_sha256"))
    if any(h is None for h in hashes) or any(h != hashes[0] for h in hashes):
        raise ValueError("Mismatching or missing test/OOD dataset SHA-256 fingerprints")
    if data["target6"]["mse"].get("no_invariance") is not False:
        raise ValueError("Expected invariant target6 MSE")
    if data["target6_noinv"]["mse"].get("no_invariance") is not True:
        raise ValueError("Expected non-invariant target6 MSE")
    original = data["original"]["mse"]
    for rec in original.get("checkpoint_provenance", {}).values():
        if rec.get("training_target_channels_zero_based") != list(range(12)):
            raise ValueError(f"Original {configuration} checkpoint is not marked as 12-target training")
    return orig_available


# =============================================================================
# Table generation
# =============================================================================

def get_cells(data: dict, spec: tuple, tex: bool) -> list[str]:
    _, _, _, metric, src, split, scale, digits = spec
    result = []
    for variant in VARIANTS:
        raw = data[variant][src]
        values = None if raw is None else get_per_seed(raw, src, split, metric)
        result.append(format_value(values, scale, digits, tex))
    return result


def tex_escape(value: str) -> str:
    # The configuration name is validated by parse_configuration.
    return value.replace("_", r"\_").replace("-", "--")


def generate_tex(data: dict, configuration: str, orig_available: bool) -> str:
    config_tex = tex_escape(configuration)
    slug = configuration.lower().replace("-", "_")
    lines = [
        "% Generated automatically by scripts.make_target_ablation_table.",
        "% Requires booktabs and graphicx.",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Effect of prediction targets and adversarial invariance.}}",
        rf"  \label{{tab:{slug}_target_ablation}}",
        r"  \small",
        r"  \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}lccc@{}}",
        r"      \toprule",
        f"      Metric & Original {config_tex} & {config_tex}--6 & {config_tex}--6 without adversarial invariance " + r"\\",
        r"      \midrule",
    ]
    last_section = None
    for spec in SPECS:
        section, _, label, *_ = spec
        if section != last_section:
            if last_section is not None:
                lines.append(r"      \addlinespace[3pt]")
            lines.append(r"      \multicolumn{4}{@{}l}{\emph{" + section + r"}} \\")
            last_section = section
        lines.append("      " + label + " & " + " & ".join(get_cells(data, spec, tex=True)) + r" \\")
    lines += [
        r"      \bottomrule",
        r"    \end{tabular}%",
        r"  }",
        r"  \par\medskip",
        r"  \begin{minipage}{\linewidth}",
        r"    \footnotesize",
        rf"    Values are mean $\pm$ sample standard deviation over five training seeds (0--4), evaluated on the ID test set unless OOD is specified. BA denotes balanced accuracy (chance: 25\%). BA gain is the signed difference between the $(S,Z)$ and $S$-only strong nuisance-class probes, in percentage points. $S$ denotes the simulated physical state and $Z$ the learned latent representation. All models receive the same 12-channel observations as input. Original {config_tex} predicts all 12 channels, whereas the {config_tex}--6 variants predict only the six nuisance-independent channels (zero-based indices 2--7). Prediction MSE is evaluated post hoc on these same six channels for all models. Restricting prediction targets also changes the effective predictive-loss normalization. The no-invariance variant disables adversarial invariance while retaining all other loss weights. Physical states are used only for post-hoc evaluation; nuisance labels are additionally used during adversarial training in the invariant configurations.",
    ]
    if not orig_available:
        lines += [
            rf"\par\emph{{Pending:}} Original {config_tex} probe metrics are not included until its",
            r"five-seed full evaluation with the same probe protocol is provided.",
            r"Dashes indicate unavailable measurements, not zero.",
        ]
    lines += [r"  \end{minipage}", r"\end{table}", ""]
    return "\n".join(lines)


def generate_markdown(data: dict, configuration: str, orig_available: bool) -> str:
    lines = [
        f"# Table S2. {configuration} prediction-target and invariance ablation",
        "",
        f"| Metric | Original {configuration} | {configuration}–6 | {configuration}–6 without invariance |",
        "|:--|--:|--:|--:|",
    ]
    last_section = None
    for spec in SPECS:
        section, label, *_ = spec
        if section != last_section:
            lines.append(f"| **{section}** | | | |")
            last_section = section
        lines.append("| " + label + " | " + " | ".join(get_cells(data, spec, tex=False)) + " |")
    lines += [
        "",
        "Mean ± sample SD over five seeds; ID test except explicitly OOD. Chance BA: 25%.",
        "BA gain = BA from (S,Z) minus BA from S, measured using the strong class probe.",
        f"Original {configuration}: 12 targets during training; other columns: six targets;",
        "all MSE values evaluated post hoc on the same six channels.",
        "Target restriction also changes the effective predictive-loss normalization.",
    ]
    if not orig_available:
        lines.append(
            f"**Incomplete draft:** original {configuration} probe values await "
            f"{configuration}-original-evaluation.json; — means unavailable."
        )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", type=parse_configuration, required=True,
                   help="Configuration ID used as a prefix for all input and output files")
    p.add_argument("--input-dir", type=Path, default=Path("runs-observable-target6"))
    p.add_argument("--output-dir", type=Path, default=Path("runs-observable-target6"))
    p.add_argument("--require-original-evaluation", action="store_true")
    args = p.parse_args()
    names = file_names(args.configuration)
    data: dict[str, dict[str, dict | None]] = {}
    for variant, paths in names.items():
        data[variant] = {}
        for typ, name in paths.items():
            path = args.input_dir / name
            if typ == "evaluation" and variant == "original" and not path.is_file():
                data[variant][typ] = None
            else:
                data[variant][typ] = load_json(path)
    orig_available = check_inputs(data, args.configuration, args.require_original_evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.configuration.lower().replace("-", "_")
    tex = args.output_dir / f"table_s2_{slug}_target_ablation.tex"
    md = args.output_dir / f"table_s2_{slug}_target_ablation.md"
    tex.write_text(generate_tex(data, args.configuration, orig_available), encoding="utf-8")
    md.write_text(generate_markdown(data, args.configuration, orig_available), encoding="utf-8")
    print(f"Wrote {tex}\nWrote {md}")
    if not orig_available:
        print(f"WARNING: Original {args.configuration} five-seed probe evaluation is missing; "
              "table cells left blank.")


if __name__ == "__main__":
    main()
