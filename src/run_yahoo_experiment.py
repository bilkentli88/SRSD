from __future__ import annotations

"""
Run the Yahoo benchmark experiments used in the paper.

This script expects locally downloaded Yahoo anomaly CSV files under a folder
such as:

    Yahoo/
        real_1.csv
        real_2.csv
        ...

It reproduces the Yahoo evaluation pipeline by:
1. loading each series,
2. computing a causal evidence stream,
3. applying the static and adaptive policies,
4. evaluating detection behavior over labeled anomaly windows,
5. exporting aggregate tables and response-inertia summaries.

The default configuration is intended to match the final Yahoo setting used
for the paper unless overridden from the command line.
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from yahoo_loader import YahooLoader
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


@dataclass(frozen=True)
class YahooExperimentConfig:
    """
    Configuration for the Yahoo benchmark evaluation.
    """

    yahoo_root: Path = Path("Yahoo")
    output_dir: Path = Path("outputs/yahoo")

    half_window_steps: int = 12
    elevated_evidence_threshold: float = 3.35
    threshold_sensitivity_values: Tuple[float, ...] = (3.0, 3.35, 3.5, 4.0)

    include_prefixes: Tuple[str, ...] = ("real_",)
    exclude_series: Tuple[str, ...] = ()
    max_series: int | None = None

    # Evidence parameters
    evidence_window: int = 48
    evidence_min_history: int = 24

    # Conservative policy parameters
    conservative_threshold: float = 4.0
    conservative_enter_count: int = 3
    conservative_exit_count: int = 3

    # Adaptive policy parameters
    adaptive_base_threshold: float = 3.35
    adaptive_exit_margin: float = 0.45
    adaptive_switch_penalty: float = 0.27


POLICY_ALARM_COLS: List[str] = [
    "alarm_aggressive",
    "alarm_mid",
    "alarm_conservative",
    "alarm_adaptive",
    "alarm_adaptive_nopenalty",
]

POLICY_DISPLAY_MAP = {
    "alarm_aggressive": "Static aggressive",
    "alarm_mid": "Static mid",
    "alarm_conservative": "Static conservative",
    "alarm_adaptive": "Adaptive",
    "alarm_adaptive_nopenalty": "Adaptive (no switch penalty)",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Yahoo-based real-data evaluation."
    )
    parser.add_argument(
        "--yahoo_root",
        type=Path,
        default=Path("Yahoo"),
        help="Path to the local Yahoo CSV folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/yahoo"),
        help="Directory where evaluation tables will be saved.",
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
        "--max_series",
        type=int,
        default=None,
        help="Optional cap on number of series to process.",
    )

    # Evidence tuning
    parser.add_argument("--evidence_window", type=int, default=48)
    parser.add_argument("--evidence_min_history", type=int, default=24)

    # Conservative tuning
    parser.add_argument("--conservative_threshold", type=float, default=4.0)
    parser.add_argument("--conservative_enter_count", type=int, default=3)
    parser.add_argument("--conservative_exit_count", type=int, default=3)

    # Adaptive tuning
    parser.add_argument("--adaptive_base_threshold", type=float, default=3.35)
    parser.add_argument("--adaptive_exit_margin", type=float, default=0.45)
    parser.add_argument("--adaptive_switch_penalty", type=float, default=0.27)

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> YahooExperimentConfig:
    """
    Build the experiment configuration from parsed CLI arguments.
    """
    cfg = YahooExperimentConfig(
        yahoo_root=args.yahoo_root,
        output_dir=args.output_dir,
    )

    if args.half_window_steps is not None:
        cfg = replace(cfg, half_window_steps=args.half_window_steps)

    if args.elevated_evidence_threshold is not None:
        cfg = replace(cfg, elevated_evidence_threshold=args.elevated_evidence_threshold)

    if args.max_series is not None:
        cfg = replace(cfg, max_series=args.max_series)

    cfg = replace(cfg, evidence_window=args.evidence_window)
    cfg = replace(cfg, evidence_min_history=args.evidence_min_history)

    cfg = replace(cfg, conservative_threshold=args.conservative_threshold)
    cfg = replace(cfg, conservative_enter_count=args.conservative_enter_count)
    cfg = replace(cfg, conservative_exit_count=args.conservative_exit_count)

    cfg = replace(cfg, adaptive_base_threshold=args.adaptive_base_threshold)
    cfg = replace(cfg, adaptive_exit_margin=args.adaptive_exit_margin)
    cfg = replace(cfg, adaptive_switch_penalty=args.adaptive_switch_penalty)

    return cfg


def make_output_dirs(base_dir: Path) -> None:
    """
    Create the output directory if it does not already exist.
    """
    base_dir.mkdir(parents=True, exist_ok=True)


def infer_category(series_name: str) -> str:
    """
    Infer a coarse category label from the filename.
    """
    if series_name.startswith("real_"):
        return "real"
    if series_name.startswith("synthetic_"):
        return "synthetic"
    return "unknown"


def run_static_mid_policy(
    df: pd.DataFrame,
    evidence_col: str = "evidence",
    threshold: float = 3.35,
) -> pd.DataFrame:
    """
    Apply a simple fixed-threshold mid-level policy.
    """
    out = df.copy()
    evidence = out[evidence_col].values

    alarm = []
    for value in evidence:
        if pd.isna(value):
            alarm.append(0)
        elif value >= threshold:
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
    Apply the adaptive policy with zero switch penalty as an ablation baseline.
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


def apply_all_policies(
    df_scored: pd.DataFrame,
    cfg: YahooExperimentConfig,
) -> pd.DataFrame:
    """
    Apply all static and adaptive policies used in the Yahoo benchmark.
    """
    out = df_scored.copy()

    out = run_aggressive_policy(
        out,
        evidence_col="evidence",
        threshold=3.0,
    )

    out = run_static_mid_policy(
        out,
        evidence_col="evidence",
        threshold=3.35,
    )

    out = run_conservative_policy(
        out,
        evidence_col="evidence",
        threshold=cfg.conservative_threshold,
        enter_count=cfg.conservative_enter_count,
        exit_count=cfg.conservative_exit_count,
    )

    out = run_adaptive_policy(
        out,
        evidence_col="evidence",
        base_threshold=cfg.adaptive_base_threshold,
        enter_margin=0.0,
        exit_margin=cfg.adaptive_exit_margin,
        adaptation_rate=0.065,
        relaxation_rate=0.04,
        switch_penalty=cfg.adaptive_switch_penalty,
        min_threshold=2.45,
        max_threshold=4.7,
    )

    out = run_adaptive_ablation_policy(
        out,
        evidence_col="evidence",
        base_threshold=cfg.adaptive_base_threshold,
        enter_margin=0.0,
        exit_margin=cfg.adaptive_exit_margin,
        adaptation_rate=0.065,
        relaxation_rate=0.04,
        min_threshold=2.45,
        max_threshold=4.7,
    )

    required_cols = POLICY_ALARM_COLS
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required policy columns: {missing}")

    return out


def collect_series_list(loader: YahooLoader, cfg: YahooExperimentConfig) -> List[str]:
    """
    Collect the list of Yahoo series to process.
    """
    series = loader.list_series()
    excluded = set(cfg.exclude_series)

    filtered = []
    for series_name in series:
        if series_name in excluded:
            continue
        if not any(series_name.startswith(prefix) for prefix in cfg.include_prefixes):
            continue
        filtered.append(series_name)

    filtered = sorted(filtered)

    if cfg.max_series is not None:
        filtered = filtered[: cfg.max_series]

    return filtered


def run_one_series(
    loader: YahooLoader,
    series_name: str,
    cfg: YahooExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full evaluation pipeline for a single Yahoo series.
    """
    df, labels = loader.load_series(series_name)

    df_scored = compute_online_robust_evidence(
        df,
        value_col="value",
        window=cfg.evidence_window,
        min_history=cfg.evidence_min_history,
        eps=1e-8,
    )

    df_scored = apply_all_policies(df_scored, cfg)

    results_df = evaluate_all_policies(
        df=df_scored,
        label_times=labels,
        policy_alarm_cols=POLICY_ALARM_COLS,
        timestamp_col="timestamp",
        half_window_steps=cfg.half_window_steps,
    )

    switch_map = {col: count_switches(df_scored[col]) for col in POLICY_ALARM_COLS}
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

    return results_df, inertia_windows_df


def build_per_series_table(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-series summary table.
    """
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
    """
    Build the policy-level aggregate table.
    """
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
    """
    Build the category-level aggregate table.
    """
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


def main() -> None:
    """
    Run the full Yahoo experiment pipeline and export result tables.
    """
    args = parse_args()
    cfg = build_config_from_args(args)
    make_output_dirs(cfg.output_dir)

    loader = YahooLoader(cfg.yahoo_root)
    series_list = collect_series_list(loader, cfg)

    print("=" * 80)
    print("RUNNING YAHOO EVALUATION")
    print("=" * 80)
    print(f"Yahoo root: {cfg.yahoo_root.resolve()}")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print(f"Number of series: {len(series_list)}")
    print(f"Evaluation window: +/- {cfg.half_window_steps} steps")
    print(f"Elevated evidence threshold: {cfg.elevated_evidence_threshold}")
    print(f"Evidence window: {cfg.evidence_window}")
    print(f"Evidence min_history: {cfg.evidence_min_history}")
    print(f"Conservative threshold: {cfg.conservative_threshold}")
    print(f"Conservative enter_count: {cfg.conservative_enter_count}")
    print(f"Conservative exit_count: {cfg.conservative_exit_count}")
    print(f"Adaptive base threshold: {cfg.adaptive_base_threshold}")
    print(f"Adaptive exit margin: {cfg.adaptive_exit_margin}")
    print(f"Adaptive switch penalty: {cfg.adaptive_switch_penalty}")
    print()

    all_result_frames: List[pd.DataFrame] = []
    all_inertia_frames: List[pd.DataFrame] = []

    for i, series_name in enumerate(series_list, start=1):
        print("-" * 80)
        print(f"[{i}/{len(series_list)}] Processing: {series_name}")

        try:
            result_df, inertia_windows_df = run_one_series(
                loader=loader,
                series_name=series_name,
                cfg=cfg,
            )

            all_result_frames.append(result_df)
            all_inertia_frames.append(inertia_windows_df)

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
            print(f"ERROR while processing {series_name}: {type(exc).__name__}: {exc}")

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
        inertia_lead_vs_nonlead_df = summarize_lead_vs_nonlead(
            all_inertia_windows_df,
            elevated_threshold=cfg.elevated_evidence_threshold,
        )
        inertia_threshold_sensitivity_df = (
            summarize_response_inertia_threshold_sensitivity(
                all_inertia_windows_df,
                thresholds=cfg.threshold_sensitivity_values,
            )
        )
    else:
        all_inertia_windows_df = pd.DataFrame()
        inertia_summary_df = pd.DataFrame()
        inertia_lead_vs_nonlead_df = pd.DataFrame()
        inertia_threshold_sensitivity_df = pd.DataFrame()

    per_series_summary_df.to_csv(
        cfg.output_dir / "yahoo_per_series_summary.csv",
        index=False,
    )
    policy_aggregate_df.to_csv(
        cfg.output_dir / "yahoo_policy_aggregate.csv",
        index=False,
    )
    category_aggregate_df.to_csv(
        cfg.output_dir / "yahoo_category_aggregate.csv",
        index=False,
    )

    all_inertia_windows_df.to_csv(
        cfg.output_dir / "yahoo_response_inertia_windows.csv",
        index=False,
    )
    inertia_summary_df.to_csv(
        cfg.output_dir / "yahoo_response_inertia_summary.csv",
        index=False,
    )
    inertia_lead_vs_nonlead_df.to_csv(
        cfg.output_dir / "yahoo_response_inertia_lead_vs_nonlead.csv",
        index=False,
    )
    inertia_threshold_sensitivity_df.to_csv(
        cfg.output_dir / "yahoo_response_inertia_threshold_sensitivity.csv",
        index=False,
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
    print("LEAD VS NON-LEAD DIAGNOSTIC")
    print("=" * 80)
    if len(inertia_lead_vs_nonlead_df) > 0:
        print(inertia_lead_vs_nonlead_df.to_string(index=False))
    else:
        print("No lead-vs-nonlead summary available.")

    print("\nSaved files:")
    print(f"- {cfg.output_dir / 'yahoo_per_series_summary.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_policy_aggregate.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_category_aggregate.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_response_inertia_windows.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_response_inertia_summary.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_response_inertia_lead_vs_nonlead.csv'}")
    print(f"- {cfg.output_dir / 'yahoo_response_inertia_threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()