"""
Run the synthetic experiment used in the paper
"On the Trade-off Between Stability and Responsiveness in Sequential Detection".

This script:
1. simulates synthetic score trajectories with a post-onset inertia-sensitive regime,
2. evaluates several static policies and one temporally adaptive policy,
3. saves raw and summary CSV files,
4. runs paired Wilcoxon tests for selected policy comparisons, and
5. generates publication-ready figures.

Outputs are written under the specified output directory with the structure:

    output_dir/
    ├── raw_csv/
    ├── summary_csv/
    └── figures/

Example:
    python src/run_synthetic_experiment.py --output_dir outputs/synthetic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Global plotting style
# ============================================================
plt.rcParams.update(
    {
        "font.size": 10,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "savefig.dpi": 600,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.autolayout": True,
    }
)


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class SyntheticConfig:
    # Timeline
    T: int = 150
    t0: int = 60

    # Pre-onset baseline
    p_pre: float = 0.38

    # Post-onset dynamics
    p_post_start: float = 0.50
    plateau_target: float = 0.55
    plateau_len: int = 20
    drift_after_plateau: float = 0.0035
    sigma: float = 0.04
    max_target: float = 0.78

    # AR(1)-style smoothing in score dynamics
    ar_coef: float = 0.75
    target_weight: float = 0.25

    # Static policies
    static_lambdas: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15)

    # Adaptive policy
    lam_high: float = 0.15
    lam_low: float = 0.05
    trigger_center: float = 0.55
    trigger_width: float = 0.16

    # Reproducibility
    n_seeds: int = 100
    base_seed: int = 2025

    # Ablation grids
    width_grid: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    center_grid: Tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)

    # Representative trajectory
    representative_seed_offset: int = 8


# ============================================================
# Utilities
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic stability-responsiveness experiment."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/synthetic"),
        help="Directory where CSV files and figures will be saved.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Override the number of random seeds used in the experiment.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Override the base random seed.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Run the experiment and save CSV files without generating figures.",
    )
    return parser.parse_args()


def make_dirs(base_dir: Path) -> Dict[str, Path]:
    raw_dir = base_dir / "raw_csv"
    summary_dir = base_dir / "summary_csv"
    figure_dir = base_dir / "figures"

    for directory in (base_dir, raw_dir, summary_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {"base": base_dir, "raw": raw_dir, "summary": summary_dir, "fig": figure_dir}


def clip_interval(center: float, width: float) -> Tuple[float, float]:
    low = center - width / 2.0
    high = center + width / 2.0
    return max(0.0, low), min(1.0, high)


# ============================================================
# Core simulation logic
# ============================================================
def post_onset_target(cfg: SyntheticConfig, t: int) -> float:
    """
    Define the post-onset score target.

    The target is designed to create an inertia-sensitive regime:
    - immediately after onset, the process enters a decision-relevant region,
    - it then lingers near that region for a plateau period,
    - after the plateau, it drifts upward more decisively.
    """
    dt = t - cfg.t0

    if dt < 0:
        return cfg.p_pre

    if dt < cfg.plateau_len:
        return cfg.plateau_target

    return min(
        cfg.max_target,
        cfg.plateau_target + cfg.drift_after_plateau * (dt - cfg.plateau_len),
    )


def simulate_switching_score(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    p = np.zeros(cfg.T, dtype=float)
    p[0] = cfg.p_pre

    for t in range(1, cfg.T):
        if t < cfg.t0:
            target = cfg.p_pre
        elif t == cfg.t0:
            target = cfg.p_post_start
        else:
            target = post_onset_target(cfg, t)

        p[t] = np.clip(
            cfg.ar_coef * p[t - 1]
            + cfg.target_weight * target
            + rng.normal(0.0, cfg.sigma),
            0.0,
            1.0,
        )

    return p


def decide_sequence_static_lambda(p: np.ndarray, lam: float) -> np.ndarray:
    T = len(p)
    y = np.zeros(T, dtype=int)

    for t in range(T):
        prev = 0 if t == 0 else y[t - 1]

        cost_if_1 = (1.0 - p[t]) + (lam if prev != 1 else 0.0)
        cost_if_0 = p[t] + (lam if prev != 0 else 0.0)

        y[t] = 1 if cost_if_1 < cost_if_0 else 0

    return y


def decide_sequence_adaptive_lambda(
    p: np.ndarray,
    lam_high: float,
    lam_low: float,
    center: float,
    width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(p)
    y = np.zeros(T, dtype=int)

    low, high = clip_interval(center, width)
    trigger = ((p >= low) & (p <= high)).astype(int)
    lam_t = np.where(trigger == 1, lam_low, lam_high)

    for t in range(T):
        prev = 0 if t == 0 else y[t - 1]
        lam = float(lam_t[t])

        cost_if_1 = (1.0 - p[t]) + (lam if prev != 1 else 0.0)
        cost_if_0 = p[t] + (lam if prev != 0 else 0.0)

        y[t] = 1 if cost_if_1 < cost_if_0 else 0

    return y, trigger, lam_t


def eval_decision_sequence(y: np.ndarray, t0: int) -> Dict[str, float]:
    post_idx = np.where(y[t0:] == 1)[0]
    delay = float(post_idx[0]) if len(post_idx) > 0 else float(len(y) - t0)

    false_alarm_onsets = 0
    prev = 0
    for t in range(t0):
        if y[t] == 1 and prev == 0:
            false_alarm_onsets += 1
        prev = y[t]

    return {
        "delay": delay,
        "false_alarm_onsets": float(false_alarm_onsets),
        "switching_count": float(np.sum(y[1:] != y[:-1])),
        "hit": float(len(post_idx) > 0),
    }


def summarize_results(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def compute_wilcoxon_tests(
    df: pd.DataFrame,
    baseline_policy: str,
    adaptive_policy: str = "Adaptive lambda_t",
    metrics: List[str] = ("delay", "switching_count"),
) -> pd.DataFrame:
    rows = []

    baseline_df = df[df["policy_name"] == baseline_policy].copy().sort_values("seed")
    adaptive_df = df[df["policy_name"] == adaptive_policy].copy().sort_values("seed")

    merged = pd.merge(
        baseline_df[["seed", *metrics]],
        adaptive_df[["seed", *metrics]],
        on="seed",
        suffixes=("_base", "_ada"),
    )

    for metric in metrics:
        x = merged[f"{metric}_base"].to_numpy()
        y = merged[f"{metric}_ada"].to_numpy()

        if np.allclose(x - y, 0):
            stat = 0.0
            p_value = np.nan
        else:
            stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

        rows.append(
            {
                "baseline_policy": baseline_policy,
                "adaptive_policy": adaptive_policy,
                "metric": metric,
                "baseline_mean": float(np.mean(x)),
                "baseline_std": float(np.std(x, ddof=1)),
                "adaptive_mean": float(np.mean(y)),
                "adaptive_std": float(np.std(y, ddof=1)),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                "n_pairs": int(len(x)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_frontier(summary_df: pd.DataFrame, output_path: Path) -> None:
    pdf_path = output_path.with_suffix(".pdf")
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    static_df = summary_df[summary_df["policy_type"] == "static"].copy()
    adaptive_df = summary_df[summary_df["policy_type"] == "adaptive"].copy()

    label_specs = {
        "Static lambda=0.00": {"offset": (8, 6), "ha": "right", "va": "bottom"},
        "Static lambda=0.05": {"offset": (3, 4), "ha": "left", "va": "bottom"},
        "Static lambda=0.10": {"offset": (5, 4), "ha": "left", "va": "bottom"},
        "Static lambda=0.15": {"offset": (3, -4), "ha": "left", "va": "bottom"},
    }

    ax.plot(
        static_df["switching_count_mean"],
        static_df["delay_mean"],
        marker="o",
        markersize=5,
        linewidth=1.2,
        label="Static policies",
        color="#1f77b4",
    )

    for _, row in static_df.iterrows():
        name = row["policy_name"]
        spec = label_specs.get(name, {"offset": (5, 5), "ha": "left", "va": "bottom"})
        annotation_text = rf"$\lambda$={row['lambda_static']:.2f}"

        ax.annotate(
            annotation_text,
            (row["switching_count_mean"], row["delay_mean"]),
            xytext=spec["offset"],
            textcoords="offset points",
            fontsize=8.5,
            ha=spec["ha"],
            va=spec["va"],
        )

    if not adaptive_df.empty:
        row = adaptive_df.iloc[0]

        ax.errorbar(
            [row["switching_count_mean"]],
            [row["delay_mean"]],
            xerr=[row["switching_count_std"]],
            yerr=[row["delay_std"]],
            fmt="x",
            markersize=8,
            elinewidth=0.7,
            capsize=2,
            label="Adaptive policy",
            color="#d62728",
            zorder=5,
        )

        ax.annotate(
            r"Adaptive $\lambda_t$",
            (row["switching_count_mean"], row["delay_mean"]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va="top",
        )

    ax.set_xlabel(r"Mean switching count $\mathbb{E}[S]$")
    ax.set_ylabel(r"Mean detection delay $\mathbb{E}[D]$")
    ax.margins(x=0.10, y=0.12)
    ax.legend(frameon=True, loc="best")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary_df: pd.DataFrame, output_path: Path, mode: str = "width") -> None:
    pdf_path = output_path.with_suffix(".pdf")
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7.5), sharex=True)

    x = summary_df["trigger_width"] if mode == "width" else summary_df["trigger_center"]
    x_label = r"Trigger width $\eta$" if mode == "width" else r"Trigger center $\gamma$"
    title = f"Trigger-{mode.capitalize()} Ablation"

    axes[0].plot(x, summary_df["delay_mean"], marker="s", markersize=4, color="#1f77b4")
    axes[0].set_ylabel(r"Mean delay $\mathbb{E}[D]$")
    axes[0].set_title(title)

    axes[1].plot(
        x,
        summary_df["false_alarm_onsets_mean"],
        marker="^",
        markersize=4,
        color="#ff7f0e",
    )
    axes[1].set_ylabel("False-alarm onsets")

    axes[2].plot(
        x,
        summary_df["switching_count_mean"],
        marker="v",
        markersize=4,
        color="#2ca02c",
    )
    axes[2].set_ylabel("Switching count")
    axes[2].set_xlabel(x_label)

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_representative_trajectory(cfg: SyntheticConfig, output_path: Path) -> None:
    """
    Plot one fixed representative trajectory for illustration.

    The seed is fixed relative to the base seed to make this figure reproducible.
    """
    seed = cfg.base_seed + cfg.representative_seed_offset
    rng = np.random.default_rng(seed)

    p = simulate_switching_score(cfg, rng)
    y_aggressive = decide_sequence_static_lambda(p, 0.00)
    y_conservative = decide_sequence_static_lambda(p, 0.15)
    y_adaptive, trigger, _ = decide_sequence_adaptive_lambda(
        p,
        cfg.lam_high,
        cfg.lam_low,
        cfg.trigger_center,
        cfg.trigger_width,
    )

    pdf_path = output_path.with_suffix(".pdf")
    t = np.arange(cfg.T)
    low, high = clip_interval(cfg.trigger_center, cfg.trigger_width)

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(t, p, label=r"Score $p_t$")
    axes[0].axvline(cfg.t0, linestyle="--", color="gray", alpha=0.7, label=r"Onset $t_0$")
    axes[0].axhspan(low, high, alpha=0.15, color="C0", label="Trigger region")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left")

    axes[1].step(t, y_aggressive, where="post", label=r"Aggressive ($\lambda=0.00$)")
    axes[1].step(t, y_conservative, where="post", label=r"Conservative ($\lambda=0.15$)")
    axes[1].set_ylabel("Alarm")
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc="upper left")

    axes[2].step(t, y_adaptive, where="post", label="Adaptive policy", linewidth=1.5)
    axes[2].step(t, trigger, where="post", linestyle="--", label="Trigger active", alpha=0.7)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("State")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="upper left")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Experiment runners
# ============================================================
def run_frontier_experiment(cfg: SyntheticConfig) -> pd.DataFrame:
    rows = []

    for seed_offset in range(cfg.n_seeds):
        seed = cfg.base_seed + seed_offset
        rng = np.random.default_rng(seed)
        p = simulate_switching_score(cfg, rng)

        for lam in cfg.static_lambdas:
            y_static = decide_sequence_static_lambda(p, lam)
            metrics = eval_decision_sequence(y_static, cfg.t0)

            rows.append(
                {
                    "experiment": "frontier",
                    "seed": seed,
                    "policy_type": "static",
                    "policy_name": f"Static lambda={lam:.2f}",
                    "lambda_static": lam,
                    "lambda_low": np.nan,
                    "lambda_high": np.nan,
                    "trigger_center": np.nan,
                    "trigger_width": np.nan,
                    **metrics,
                }
            )

        y_adaptive, _, _ = decide_sequence_adaptive_lambda(
            p,
            cfg.lam_high,
            cfg.lam_low,
            cfg.trigger_center,
            cfg.trigger_width,
        )
        rows.append(
            {
                "experiment": "frontier",
                "seed": seed,
                "policy_type": "adaptive",
                "policy_name": "Adaptive lambda_t",
                "lambda_static": np.nan,
                "lambda_low": cfg.lam_low,
                "lambda_high": cfg.lam_high,
                "trigger_center": cfg.trigger_center,
                "trigger_width": cfg.trigger_width,
                **eval_decision_sequence(y_adaptive, cfg.t0),
            }
        )

    return pd.DataFrame(rows)


def run_width_ablation(cfg: SyntheticConfig) -> pd.DataFrame:
    rows = []

    for width in cfg.width_grid:
        for seed_offset in range(cfg.n_seeds):
            rng = np.random.default_rng(cfg.base_seed + seed_offset)
            p = simulate_switching_score(cfg, rng)
            y, _, _ = decide_sequence_adaptive_lambda(
                p,
                cfg.lam_high,
                cfg.lam_low,
                cfg.trigger_center,
                width,
            )
            rows.append(
                {
                    "trigger_width": width,
                    **eval_decision_sequence(y, cfg.t0),
                }
            )

    return pd.DataFrame(rows)


def run_location_ablation(cfg: SyntheticConfig) -> pd.DataFrame:
    rows = []

    for center in cfg.center_grid:
        for seed_offset in range(cfg.n_seeds):
            rng = np.random.default_rng(cfg.base_seed + seed_offset)
            p = simulate_switching_score(cfg, rng)
            y, _, _ = decide_sequence_adaptive_lambda(
                p,
                cfg.lam_high,
                cfg.lam_low,
                center,
                cfg.trigger_width,
            )
            rows.append(
                {
                    "trigger_center": center,
                    **eval_decision_sequence(y, cfg.t0),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    cfg = SyntheticConfig()
    if args.n_seeds is not None:
        cfg = replace(cfg, n_seeds=args.n_seeds)
    if args.base_seed is not None:
        cfg = replace(cfg, base_seed=args.base_seed)

    dirs = make_dirs(args.output_dir)

    # Frontier experiment
    frontier_raw_df = run_frontier_experiment(cfg)
    frontier_raw_df.to_csv(dirs["raw"] / "frontier_raw.csv", index=False)

    frontier_summary_df = summarize_results(
        frontier_raw_df,
        [
            "policy_type",
            "policy_name",
            "lambda_static",
            "lambda_low",
            "lambda_high",
            "trigger_center",
            "trigger_width",
        ],
        ["delay", "false_alarm_onsets", "switching_count"],
    )
    frontier_summary_df.to_csv(dirs["summary"] / "frontier_summary.csv", index=False)

    # Comparison against a moderate static baseline
    wilcoxon_vs_static_005_df = compute_wilcoxon_tests(
        frontier_raw_df,
        baseline_policy="Static lambda=0.05",
        adaptive_policy="Adaptive lambda_t",
        metrics=["delay", "switching_count"],
    )
    wilcoxon_vs_static_005_df.to_csv(
        dirs["summary"] / "frontier_wilcoxon_vs_static005.csv",
        index=False,
    )

    # Comparison against the conservative static baseline
    wilcoxon_vs_static_015_df = compute_wilcoxon_tests(
        frontier_raw_df,
        baseline_policy="Static lambda=0.15",
        adaptive_policy="Adaptive lambda_t",
        metrics=["delay", "switching_count"],
    )
    wilcoxon_vs_static_015_df.to_csv(
        dirs["summary"] / "frontier_wilcoxon_vs_static015.csv",
        index=False,
    )

    # Width ablation
    width_ablation_raw_df = run_width_ablation(cfg)
    width_ablation_raw_df.to_csv(
        dirs["raw"] / "trigger_width_ablation_raw.csv",
        index=False,
    )

    width_ablation_summary_df = summarize_results(
        width_ablation_raw_df,
        ["trigger_width"],
        ["delay", "false_alarm_onsets", "switching_count"],
    )
    width_ablation_summary_df.to_csv(
        dirs["summary"] / "trigger_width_ablation_summary.csv",
        index=False,
    )

    # Location ablation
    location_ablation_raw_df = run_location_ablation(cfg)
    location_ablation_raw_df.to_csv(
        dirs["raw"] / "trigger_location_ablation_raw.csv",
        index=False,
    )

    location_ablation_summary_df = summarize_results(
        location_ablation_raw_df,
        ["trigger_center"],
        ["delay", "false_alarm_onsets", "switching_count"],
    )
    location_ablation_summary_df.to_csv(
        dirs["summary"] / "trigger_location_ablation_summary.csv",
        index=False,
    )

    if not args.skip_plots:
        plot_frontier(frontier_summary_df, dirs["fig"] / "frontier_delay_vs_switching")
        plot_ablation(
            width_ablation_summary_df,
            dirs["fig"] / "trigger_width_ablation",
            mode="width",
        )
        plot_ablation(
            location_ablation_summary_df,
            dirs["fig"] / "trigger_location_ablation",
            mode="location",
        )
        plot_representative_trajectory(
            cfg,
            dirs["fig"] / "representative_trajectory",
        )

    print(f"Synthetic study finished. Outputs saved under: {dirs['base'].resolve()}")


if __name__ == "__main__":
    main()