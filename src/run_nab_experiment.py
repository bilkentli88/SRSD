"""
Run the NAB-based real-data evaluation used in the paper
"On the Trade-off Between Stability and Responsiveness in Sequential Detection".

This script:
1. loads selected NAB time series,
2. computes online robust evidence,
3. evaluates static and adaptive policies,
4. summarizes policy-level and category-level results,
5. computes response-inertia diagnostics, and
6. saves tables and case-study plots.

The repository is intended to reproduce both main-paper and supplement artifacts.
Accordingly, this script retains case-study and diagnostic outputs that support
either the main manuscript or the supplementary material.

Expected companion modules in src/:
    - nab_loader.py
    - evidence.py
    - policies.py
    - metrics.py
    - evaluation.py
    - response_inertia_analysis.py

Example:
    python src/run_nab_experiment.py --nab_root data/NAB --output_dir outputs/nab
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from nab_loader import NABLoader
from evidence import compute_online_robust_evidence
from policies import (
    run_aggressive_policy,
    run_conservative_policy,
    run_adaptive_policy,
)
from metrics import count_switches
from evaluation import evaluate_all_policies
from response_inertia_analysis import (
    compare_conservative_vs_adaptive_windows,
    summarize_response_inertia,
    summarize_lead_vs_nonlead,
    summarize_response_inertia_threshold_sensitivity,
)


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class NABExperimentConfig:
    nab_root: Path = Path("data/NAB")
    output_dir: Path = Path("outputs/nab")

    half_window_steps: int = 24
    elevated_evidence_threshold: float = 3.35
    threshold_sensitivity_values: Tuple[float, ...] = (3.0, 3.35, 3.5, 4.0)

    category_prefixes: Tuple[str, ...] = (
        "realKnownCause/",
        "realTraffic/",
        "realTweets/",
        "realAWSCloudwatch/",
    )

    exclude_series: Tuple[str, ...] = ()

    # The repository reproduces both main-paper and supplement artifacts.
    case_study_series: Tuple[str, ...] = (
        "realKnownCause/ec2_request_latency_system_failure.csv",
        "realTraffic/TravelTime_451.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NAB-based real-data evaluation."
    )
    parser.add_argument(
        "--nab_root",
        type=Path,
        default=Path("data/NAB"),
        help="Path to the NAB dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/nab"),
        help="Directory where evaluation tables and figures will be saved.",
    )
    parser.add_argument(
        "--half_window_steps",
        type=int,
        default=None,
        help="Override the half-window size used around labeled events.",
    )
    parser.add_argument(
        "--elevated_evidence_threshold",
        type=float,
        default=None,
        help="Override the elevated-evidence threshold used in response-inertia diagnostics.",
    )
    parser.add_argument(
        "--skip_case_plots",
        action="store_true",
        help="Run the full evaluation but skip case-study plot generation.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> NABExperimentConfig:
    cfg = NABExperimentConfig(
        nab_root=args.nab_root,
        output_dir=args.output_dir,
    )

    if args.half_window_steps is not None:
        cfg = replace(cfg, half_window_steps=args.half_window_steps)

    if args.elevated_evidence_threshold is not None:
        cfg = replace(
            cfg,
            elevated_evidence_threshold=args.elevated_evidence_threshold,
        )

    return cfg


def make_output_dirs(base_dir: Path) -> Dict[str, Path]:
    figures_dir = base_dir / "figures"

    for directory in (base_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {"base": base_dir, "figures": figures_dir}


# ============================================================
# Policy helpers
# ============================================================

def run_mid_policy(
    df: pd.DataFrame,
    evidence_col: str = "evidence",
    threshold: float = 3.35,
) -> pd.DataFrame:
    """Apply a simple static mid-threshold baseline."""
    out = df.copy()
    evidence = out[evidence_col].values

    alarm = []
    for e in evidence:
        if pd.isna(e):
            alarm.append(0)
        elif e >= threshold:
            alarm.append(1)
        else:
            alarm.append(0)

    out["alarm_mid"] = alarm
    return out


def run_adaptive_ablation_policy(
    df: pd.DataFrame,
    evidence_col: str = "evidence",
    base_threshold: float = 3.35,
    enter_margin: float = 0.0,
    exit_margin: float = 0.45,
    adaptation_rate: float = 0.065,
    relaxation_rate: float = 0.04,
    min_threshold: float = 2.45,
    max_threshold: float = 4.7,
) -> pd.DataFrame:
    """
    Apply an adaptive threshold policy without switching penalty.

    This ablation is computed on a temporary copy so that it does not overwrite
    the main adaptive-policy columns.
    """
    temp = run_adaptive_policy(
        df=df.copy(),
        evidence_col=evidence_col,
        base_threshold=base_threshold,
        enter_margin=enter_margin,
        exit_margin=exit_margin,
        adaptation_rate=adaptation_rate,
        relaxation_rate=relaxation_rate,
        switch_penalty=0.0,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

    out = df.copy()
    out["alarm_adaptive_nopenalty"] = temp["alarm_adaptive"].values
    out["adaptive_threshold_nopenalty"] = temp["adaptive_threshold"].values
    return out


def apply_all_policies(df_scored: pd.DataFrame) -> pd.DataFrame:
    out = df_scored.copy()

    out = run_aggressive_policy(
        out,
        evidence_col="evidence",
        threshold=3.0,
    )

    out = run_mid_policy(
        out,
        evidence_col="evidence",
        threshold=3.35,
    )

    out = run_conservative_policy(
        out,
        evidence_col="evidence",
        threshold=4.0,
        enter_count=3,
        exit_count=3,
    )

    out = run_adaptive_policy(
        out,
        evidence_col="evidence",
        base_threshold=3.35,
        enter_margin=0.0,
        exit_margin=0.45,
        adaptation_rate=0.065,
        relaxation_rate=0.04,
        switch_penalty=0.27,
        min_threshold=2.45,
        max_threshold=4.7,
    )

    out = run_adaptive_ablation_policy(
        out,
        evidence_col="evidence",
        base_threshold=3.35,
        enter_margin=0.0,
        exit_margin=0.45,
        adaptation_rate=0.065,
        relaxation_rate=0.04,
        min_threshold=2.45,
        max_threshold=4.7,
    )

    required_cols = [
        "alarm_aggressive",
        "alarm_mid",
        "alarm_conservative",
        "alarm_adaptive",
        "alarm_adaptive_nopenalty",
    ]
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required policy columns: {missing}")

    return out


POLICY_ALARM_COLS: List[str] = [
    "alarm_aggressive",
    "alarm_mid",
    "alarm_conservative",
    "alarm_adaptive",
    "alarm_adaptive_nopenalty",
]


POLICY_DISPLAY_MAP: Dict[str, str] = {
    "alarm_aggressive": "Static aggressive",
    "alarm_mid": "Static mid",
    "alarm_conservative": "Static conservative",
    "alarm_adaptive": "Adaptive",
    "alarm_adaptive_nopenalty": "Adaptive (no switch penalty)",
}


# ============================================================
# Series helpers
# ============================================================

def infer_category(series_name: str) -> str:
    return series_name.split("/")[0]


def collect_series_list(loader: NABLoader, cfg: NABExperimentConfig) -> List[str]:
    series: List[str] = []
    excluded = set(cfg.exclude_series)

    for prefix in cfg.category_prefixes:
        for series_name in loader.list_series(prefix=prefix):
            if series_name not in excluded:
                series.append(series_name)

    return sorted(series)


def run_one_series(
    loader: NABLoader,
    series_name: str,
    cfg: NABExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[pd.Timestamp], pd.DataFrame]:
    df, labels = loader.load_series(series_name)

    df_scored = compute_online_robust_evidence(
        df,
        value_col="value",
        window=48,
        min_history=24,
        eps=1e-8,
    )

    df_scored = apply_all_policies(df_scored)

    results_df = evaluate_all_policies(
        df=df_scored,
        label_times=labels,
        policy_alarm_cols=POLICY_ALARM_COLS,
        timestamp_col="timestamp",
        half_window_steps=cfg.half_window_steps,
    )

    switch_map = {
        col: count_switches(df_scored[col]) for col in POLICY_ALARM_COLS
    }
    results_df["switches"] = results_df["alarm_col"].map(switch_map)
    results_df["series"] = series_name
    results_df["category"] = infer_category(series_name)
    results_df["policy_name"] = results_df["alarm_col"].map(POLICY_DISPLAY_MAP)

    inertia_windows_df = compare_conservative_vs_adaptive_windows(
        df=df_scored,
        label_times=labels,
        timestamp_col="timestamp",
        evidence_col="evidence",
        conservative_alarm_col="alarm_conservative",
        adaptive_alarm_col="alarm_adaptive",
        adaptive_threshold_col="adaptive_threshold",
        half_window_steps=cfg.half_window_steps,
        elevated_evidence_threshold=cfg.elevated_evidence_threshold,
    )

    inertia_windows_df["series"] = series_name
    inertia_windows_df["category"] = infer_category(series_name)

    return results_df, df_scored, labels, inertia_windows_df


# ============================================================
# Aggregation
# ============================================================

def build_per_series_table(all_results: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "category",
        "series",
        "alarm_col",
        "policy_name",
        "n_labels",
        "hits",
        "misses",
        "hit_rate",
        "mean_delay_steps",
        "median_delay_steps",
        "mean_delay_minutes",
        "false_alarm_onsets",
        "total_alarm_onsets",
        "switches",
    ]
    return all_results[keep_cols].copy()


def build_policy_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        summary_df.groupby(["alarm_col", "policy_name"], as_index=False)
        .agg(
            {
                "n_labels": "sum",
                "hits": "sum",
                "misses": "sum",
                "false_alarm_onsets": "sum",
                "total_alarm_onsets": "sum",
                "switches": "sum",
            }
        )
    )

    macro = (
        summary_df.groupby(["alarm_col", "policy_name"], as_index=False)
        .agg(
            {
                "hit_rate": "mean",
                "mean_delay_steps": "mean",
                "median_delay_steps": "mean",
                "mean_delay_minutes": "mean",
                "false_alarm_onsets": "mean",
                "total_alarm_onsets": "mean",
                "switches": "mean",
            }
        )
        .rename(
            columns={
                "hit_rate": "mean_hit_rate_across_series",
                "mean_delay_steps": "mean_delay_steps_across_series",
                "median_delay_steps": "median_delay_steps_across_series",
                "mean_delay_minutes": "mean_delay_minutes_across_series",
                "false_alarm_onsets": "mean_false_alarm_onsets_across_series",
                "total_alarm_onsets": "mean_total_alarm_onsets_across_series",
                "switches": "mean_switches_across_series",
            }
        )
    )

    out = totals.merge(macro, on=["alarm_col", "policy_name"], how="left")
    out["micro_hit_rate"] = out["hits"] / out["n_labels"]
    return out


def build_category_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        summary_df.groupby(["category", "alarm_col", "policy_name"], as_index=False)
        .agg(
            {
                "n_labels": "sum",
                "hits": "sum",
                "misses": "sum",
                "false_alarm_onsets": "sum",
                "total_alarm_onsets": "sum",
                "switches": "sum",
                "hit_rate": "mean",
                "mean_delay_minutes": "mean",
            }
        )
        .rename(
            columns={
                "hit_rate": "mean_hit_rate_across_series",
                "mean_delay_minutes": "mean_delay_minutes_across_series",
            }
        )
    )
    totals["micro_hit_rate"] = totals["hits"] / totals["n_labels"]
    return totals


def build_inertia_category_aggregate(all_inertia_windows: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, group in all_inertia_windows.groupby("category"):
        summary = summarize_response_inertia(group).iloc[0].to_dict()
        summary["category"] = category
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    cols = ["category"] + [col for col in rows[0].keys() if col != "category"]
    return pd.DataFrame(rows)[cols]


# ============================================================
# Plotting
# ============================================================

def build_label_windows_for_plot(
    label_times: Sequence[pd.Timestamp],
    timestamps: Sequence[pd.Timestamp],
    half_window_steps: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    timestamps_series = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)

    if len(timestamps_series) < 2:
        return []

    diffs = timestamps_series.diff().dropna()
    step = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.median()
    delta = half_window_steps * step

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for t in pd.to_datetime(label_times):
        windows.append((t - delta, t + delta))
    return windows


def plot_case_study(
    series_name: str,
    df_scored: pd.DataFrame,
    label_times: Sequence[pd.Timestamp],
    output_path: Path,
    half_window_steps: int = 24,
) -> None:
    timestamps = pd.to_datetime(df_scored["timestamp"])
    windows = build_label_windows_for_plot(label_times, timestamps, half_window_steps)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 1: raw time series
    axes[0].plot(timestamps, df_scored["value"], label="Value")
    for start, end in windows:
        axes[0].axvspan(start, end, alpha=0.18)
    axes[0].set_ylabel("Value")
    axes[0].set_title(series_name)

    # Panel 2: evidence and adaptive threshold
    axes[1].plot(timestamps, df_scored["evidence"], label="Evidence")
    if "adaptive_threshold" in df_scored.columns:
        axes[1].plot(
            timestamps,
            df_scored["adaptive_threshold"],
            label="Adaptive threshold",
        )
    for start, end in windows:
        axes[1].axvspan(start, end, alpha=0.18)
    axes[1].set_ylabel("Evidence")
    axes[1].legend(loc="upper right")

    # Panel 3: alarms
    offset_map = {
        "alarm_aggressive": 4,
        "alarm_mid": 3,
        "alarm_conservative": 2,
        "alarm_adaptive": 1,
        "alarm_adaptive_nopenalty": 0,
    }
    for col, offset in offset_map.items():
        y = df_scored[col].astype(int).values
        axes[2].step(
            timestamps,
            y + offset,
            where="post",
            label=POLICY_DISPLAY_MAP[col],
        )

    for start, end in windows:
        axes[2].axvspan(start, end, alpha=0.18)

    axes[2].set_ylabel("Alarm bands")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right", ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    dirs = make_output_dirs(cfg.output_dir)

    loader = NABLoader(cfg.nab_root)
    series_list = collect_series_list(loader, cfg)

    print("=" * 80)
    print("RUNNING NAB EVALUATION")
    print("=" * 80)
    print(f"NAB root: {cfg.nab_root.resolve()}")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print(f"Number of series: {len(series_list)}")
    print("Included categories:")
    for prefix in cfg.category_prefixes:
        print(f"- {prefix}")
    print(f"Evaluation window: +/- {cfg.half_window_steps} steps")
    print(f"Elevated evidence threshold: {cfg.elevated_evidence_threshold}")
    print(f"Threshold sensitivity values: {cfg.threshold_sensitivity_values}")
    print()

    all_result_frames: List[pd.DataFrame] = []
    all_inertia_frames: List[pd.DataFrame] = []
    cached_case_studies: Dict[str, Tuple[pd.DataFrame, Sequence[pd.Timestamp]]] = {}

    for i, series_name in enumerate(series_list, start=1):
        print("-" * 80)
        print(f"[{i}/{len(series_list)}] Processing: {series_name}")

        try:
            result_df, df_scored, labels, inertia_windows_df = run_one_series(
                loader=loader,
                series_name=series_name,
                cfg=cfg,
            )

            all_result_frames.append(result_df)
            all_inertia_frames.append(inertia_windows_df)

            if series_name in cfg.case_study_series:
                cached_case_studies[series_name] = (df_scored.copy(), labels.copy())

            print(
                result_df[
                    [
                        "policy_name",
                        "hits",
                        "misses",
                        "hit_rate",
                        "mean_delay_minutes",
                        "false_alarm_onsets",
                        "switches",
                    ]
                ].to_string(index=False)
            )

        except Exception as exc:
            print(f"ERROR while processing {series_name}: {exc}")

        print()

    if not all_result_frames:
        print("No series were processed successfully.")
        return

    all_results_df = pd.concat(all_result_frames, ignore_index=True)
    per_series_summary_df = build_per_series_table(all_results_df)
    policy_aggregate_df = build_policy_aggregate(per_series_summary_df)
    category_aggregate_df = build_category_aggregate(per_series_summary_df)

    if all_inertia_frames:
        all_inertia_windows_df = pd.concat(all_inertia_frames, ignore_index=True)

        inertia_summary_df = summarize_response_inertia(all_inertia_windows_df)
        inertia_by_category_df = build_inertia_category_aggregate(all_inertia_windows_df)

        inertia_lead_vs_nonlead_df = summarize_lead_vs_nonlead(
            all_inertia_windows_df,
            elevated_threshold=cfg.elevated_evidence_threshold,
        )

        inertia_threshold_sensitivity_df = summarize_response_inertia_threshold_sensitivity(
            all_inertia_windows_df,
            thresholds=cfg.threshold_sensitivity_values,
        )
    else:
        all_inertia_windows_df = pd.DataFrame()
        inertia_summary_df = pd.DataFrame()
        inertia_by_category_df = pd.DataFrame()
        inertia_lead_vs_nonlead_df = pd.DataFrame()
        inertia_threshold_sensitivity_df = pd.DataFrame()

    # Save tables
    per_series_summary_csv = cfg.output_dir / "nab_per_series_summary.csv"
    policy_aggregate_csv = cfg.output_dir / "nab_policy_aggregate.csv"
    category_aggregate_csv = cfg.output_dir / "nab_category_aggregate.csv"

    inertia_windows_csv = cfg.output_dir / "nab_response_inertia_windows.csv"
    inertia_summary_csv = cfg.output_dir / "nab_response_inertia_summary.csv"
    inertia_by_category_csv = cfg.output_dir / "nab_response_inertia_by_category.csv"
    inertia_lead_vs_nonlead_csv = cfg.output_dir / "nab_response_inertia_lead_vs_nonlead.csv"
    inertia_threshold_sensitivity_csv = (
        cfg.output_dir / "nab_response_inertia_threshold_sensitivity.csv"
    )

    per_series_summary_df.to_csv(per_series_summary_csv, index=False)
    policy_aggregate_df.to_csv(policy_aggregate_csv, index=False)
    category_aggregate_df.to_csv(category_aggregate_csv, index=False)

    all_inertia_windows_df.to_csv(inertia_windows_csv, index=False)
    inertia_summary_df.to_csv(inertia_summary_csv, index=False)
    inertia_by_category_df.to_csv(inertia_by_category_csv, index=False)
    inertia_lead_vs_nonlead_df.to_csv(inertia_lead_vs_nonlead_csv, index=False)
    inertia_threshold_sensitivity_df.to_csv(
        inertia_threshold_sensitivity_csv,
        index=False,
    )

    # Save case-study plots
    if not args.skip_case_plots:
        for series_name, (df_scored, labels) in cached_case_studies.items():
            safe_name = series_name.replace("/", "__").replace(".csv", "")
            plot_case_study(
                series_name=series_name,
                df_scored=df_scored,
                label_times=labels,
                output_path=dirs["figures"] / f"{safe_name}_case_study.png",
                half_window_steps=cfg.half_window_steps,
            )

    print("=" * 80)
    print("POLICY-LEVEL AGGREGATE")
    print("=" * 80)
    print(policy_aggregate_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("CATEGORY-LEVEL AGGREGATE")
    print("=" * 80)
    print(category_aggregate_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("RESPONSE-INERTIA SUMMARY")
    print("=" * 80)
    if len(inertia_summary_df) > 0:
        print(inertia_summary_df.to_string(index=False))
    else:
        print("No response-inertia summary available.")

    print("\n" + "=" * 80)
    print("RESPONSE-INERTIA BY CATEGORY")
    print("=" * 80)
    if len(inertia_by_category_df) > 0:
        print(inertia_by_category_df.to_string(index=False))
    else:
        print("No category-level response-inertia summary available.")

    print("\n" + "=" * 80)
    print("LEAD VS NON-LEAD DIAGNOSTIC")
    print("=" * 80)
    if len(inertia_lead_vs_nonlead_df) > 0:
        print(inertia_lead_vs_nonlead_df.to_string(index=False))
    else:
        print("No lead-vs-nonlead diagnostic available.")

    print("\n" + "=" * 80)
    print("THRESHOLD-SENSITIVITY DIAGNOSTIC")
    print("=" * 80)
    if len(inertia_threshold_sensitivity_df) > 0:
        print(inertia_threshold_sensitivity_df.to_string(index=False))
    else:
        print("No threshold-sensitivity diagnostic available.")

    print("\nSaved files:")
    print(per_series_summary_csv)
    print(policy_aggregate_csv)
    print(category_aggregate_csv)
    print(inertia_windows_csv)
    print(inertia_summary_csv)
    print(inertia_by_category_csv)
    print(inertia_lead_vs_nonlead_csv)
    print(inertia_threshold_sensitivity_csv)
    if not args.skip_case_plots:
        print(dirs["figures"])


if __name__ == "__main__":
    main()