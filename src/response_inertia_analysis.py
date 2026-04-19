"""
Response-inertia analysis utilities for the NAB-based real-data study.

This module compares conservative and adaptive policies on a per-label-window
basis and produces summary tables used in the paper and supplementary material.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact as scipy_fisher_exact
except Exception:
    scipy_fisher_exact = None


def infer_time_step(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Timedelta:
    """
    Infer the dominant time step of a timestamped series.
    """
    timestamps = pd.to_datetime(df[timestamp_col])
    diffs = timestamps.diff().dropna()

    if len(diffs) == 0:
        raise ValueError("Cannot infer time step from fewer than 2 timestamps.")

    mode_vals = diffs.mode()
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]

    return diffs.median()


def build_label_windows(
    label_times: Sequence[pd.Timestamp],
    step: pd.Timedelta,
    half_window_steps: int = 24,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build symmetric evaluation windows around labeled event times.

    Returns
    -------
    list of tuples
        Each tuple is (label_time, window_start, window_end).
    """
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    delta = half_window_steps * step

    for t in pd.to_datetime(label_times):
        start = t - delta
        end = t + delta
        windows.append((pd.Timestamp(t), pd.Timestamp(start), pd.Timestamp(end)))

    return windows


def _first_alarm_index_in_window(
    timestamps: pd.Series,
    alarm: np.ndarray,
    label_time: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Dict[str, object]:
    """
    Identify the first alarm occurrence relevant to a single label window.

    Returns a dictionary with:
      - hit: whether the policy is active somewhere in the window
      - active_at_label: whether the policy is already active at the label time
      - first_alarm_idx: first relevant alarm index within the window
      - first_alarm_time: timestamp of that alarm
      - delay_steps: 0 if already active at the label boundary, otherwise NaN
        to be filled later when the time step is available
    """
    in_window_mask = (timestamps >= window_start) & (timestamps <= window_end)
    in_window_indices = np.where(in_window_mask)[0]

    if len(in_window_indices) == 0:
        return {
            "hit": False,
            "active_at_label": False,
            "first_alarm_idx": None,
            "first_alarm_time": pd.NaT,
            "delay_steps": np.nan,
        }

    alarmed_indices = [idx for idx in in_window_indices if alarm[idx] == 1]
    if len(alarmed_indices) == 0:
        return {
            "hit": False,
            "active_at_label": False,
            "first_alarm_idx": None,
            "first_alarm_time": pd.NaT,
            "delay_steps": np.nan,
        }

    label_or_after = [idx for idx in in_window_indices if timestamps.iloc[idx] >= label_time]
    prev_indices = [idx for idx in in_window_indices if timestamps.iloc[idx] < label_time]

    # Already active exactly at the label time.
    for idx in label_or_after:
        if timestamps.iloc[idx] == label_time and alarm[idx] == 1:
            return {
                "hit": True,
                "active_at_label": True,
                "first_alarm_idx": idx,
                "first_alarm_time": timestamps.iloc[idx],
                "delay_steps": 0.0,
            }

    # Already active across the label boundary.
    if len(prev_indices) > 0 and len(label_or_after) > 0:
        last_before = prev_indices[-1]
        first_after = label_or_after[0]
        if alarm[last_before] == 1 and alarm[first_after] == 1:
            return {
                "hit": True,
                "active_at_label": True,
                "first_alarm_idx": first_after,
                "first_alarm_time": timestamps.iloc[first_after],
                "delay_steps": 0.0,
            }

    # First active point at or after the label.
    after_alarm_indices = [idx for idx in label_or_after if alarm[idx] == 1]
    if len(after_alarm_indices) > 0:
        first_idx = after_alarm_indices[0]
        return {
            "hit": True,
            "active_at_label": False,
            "first_alarm_idx": first_idx,
            "first_alarm_time": timestamps.iloc[first_idx],
            "delay_steps": np.nan,
        }

    # Fallback: only pre-label alarm exists within the window.
    fallback_idx = alarmed_indices[0]
    return {
        "hit": True,
        "active_at_label": True,
        "first_alarm_idx": fallback_idx,
        "first_alarm_time": timestamps.iloc[fallback_idx],
        "delay_steps": 0.0,
    }


def _safe_scalar(x: object) -> float:
    """
    Convert a scalar-like value to float, returning NaN when unavailable.
    """
    if x is None:
        return np.nan
    if pd.isna(x):
        return np.nan
    return float(x)


def compare_conservative_vs_adaptive_windows(
    df: pd.DataFrame,
    label_times: Sequence[pd.Timestamp],
    timestamp_col: str = "timestamp",
    evidence_col: str = "evidence",
    conservative_alarm_col: str = "alarm_conservative",
    adaptive_alarm_col: str = "alarm_adaptive",
    adaptive_threshold_col: str = "adaptive_threshold",
    half_window_steps: int = 24,
    elevated_evidence_threshold: float = 3.35,
) -> pd.DataFrame:
    """
    Build a per-window comparison between conservative and adaptive policies.

    A window is marked as an inertial candidate when:
      - the adaptive policy detects earlier than the conservative policy, and
      - evidence was already elevated before the conservative policy acted.
    """
    out_rows: List[Dict[str, object]] = []

    timestamps = pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
    evidence = pd.to_numeric(df[evidence_col], errors="coerce").reset_index(drop=True)
    conservative_alarm = np.asarray(df[conservative_alarm_col], dtype=int)
    adaptive_alarm = np.asarray(df[adaptive_alarm_col], dtype=int)

    if adaptive_threshold_col in df.columns:
        adaptive_threshold = pd.to_numeric(
            df[adaptive_threshold_col],
            errors="coerce",
        ).reset_index(drop=True)
    else:
        adaptive_threshold = pd.Series(np.nan, index=df.index)

    step = infer_time_step(df, timestamp_col=timestamp_col)
    step_minutes = float(step / pd.Timedelta(minutes=1))

    label_windows = build_label_windows(
        label_times=label_times,
        step=step,
        half_window_steps=half_window_steps,
    )

    for label_idx, (label_time, window_start, window_end) in enumerate(label_windows, start=1):
        conservative_info = _first_alarm_index_in_window(
            timestamps=timestamps,
            alarm=conservative_alarm,
            label_time=label_time,
            window_start=window_start,
            window_end=window_end,
        )
        adaptive_info = _first_alarm_index_in_window(
            timestamps=timestamps,
            alarm=adaptive_alarm,
            label_time=label_time,
            window_start=window_start,
            window_end=window_end,
        )

        if (
            conservative_info["hit"]
            and np.isnan(conservative_info["delay_steps"])
            and conservative_info["first_alarm_idx"] is not None
        ):
            conservative_info["delay_steps"] = float(
                (conservative_info["first_alarm_time"] - label_time) / step
            )

        if (
            adaptive_info["hit"]
            and np.isnan(adaptive_info["delay_steps"])
            and adaptive_info["first_alarm_idx"] is not None
        ):
            adaptive_info["delay_steps"] = float(
                (adaptive_info["first_alarm_time"] - label_time) / step
            )

        conservative_delay = _safe_scalar(conservative_info["delay_steps"])
        adaptive_delay = _safe_scalar(adaptive_info["delay_steps"])

        if np.isnan(conservative_delay) or np.isnan(adaptive_delay):
            adaptive_lead_steps = np.nan
        else:
            adaptive_lead_steps = max(conservative_delay - adaptive_delay, 0.0)

        adaptive_leads = bool(
            not np.isnan(adaptive_lead_steps) and adaptive_lead_steps > 0
        )

        if conservative_info["first_alarm_idx"] is not None:
            pre_cons_mask = (
                (timestamps >= label_time)
                & (timestamps < conservative_info["first_alarm_time"])
            )
        else:
            pre_cons_mask = (timestamps >= label_time) & (timestamps <= window_end)

        pre_cons_evidence = evidence[pre_cons_mask].dropna()

        if len(pre_cons_evidence) > 0:
            max_evidence_pre_cons = float(pre_cons_evidence.max())
            mean_evidence_pre_cons = float(pre_cons_evidence.mean())
            elevated_evidence_flag = bool(
                max_evidence_pre_cons >= elevated_evidence_threshold
            )
            elevated_evidence_count = int(
                (pre_cons_evidence >= elevated_evidence_threshold).sum()
            )
        else:
            max_evidence_pre_cons = np.nan
            mean_evidence_pre_cons = np.nan
            elevated_evidence_flag = False
            elevated_evidence_count = 0

        if conservative_info["first_alarm_idx"] is not None:
            pre_cons_adaptive_threshold = adaptive_threshold[pre_cons_mask].dropna()
            mean_adaptive_threshold_pre_cons = (
                float(pre_cons_adaptive_threshold.mean())
                if len(pre_cons_adaptive_threshold) > 0
                else np.nan
            )
            min_adaptive_threshold_pre_cons = (
                float(pre_cons_adaptive_threshold.min())
                if len(pre_cons_adaptive_threshold) > 0
                else np.nan
            )
        else:
            mean_adaptive_threshold_pre_cons = np.nan
            min_adaptive_threshold_pre_cons = np.nan

        out_rows.append(
            {
                "label_id": int(label_idx),
                "label_time": pd.Timestamp(label_time),
                "window_start": pd.Timestamp(window_start),
                "window_end": pd.Timestamp(window_end),
                "cons_hit": bool(conservative_info["hit"]),
                "adapt_hit": bool(adaptive_info["hit"]),
                "cons_active_at_label": bool(conservative_info["active_at_label"]),
                "adapt_active_at_label": bool(adaptive_info["active_at_label"]),
                "cons_first_alarm_time": conservative_info["first_alarm_time"],
                "adapt_first_alarm_time": adaptive_info["first_alarm_time"],
                "cons_delay_steps": conservative_delay,
                "adapt_delay_steps": adaptive_delay,
                "cons_delay_minutes": (
                    conservative_delay * step_minutes
                    if not np.isnan(conservative_delay)
                    else np.nan
                ),
                "adapt_delay_minutes": (
                    adaptive_delay * step_minutes
                    if not np.isnan(adaptive_delay)
                    else np.nan
                ),
                "adaptive_lead_steps": adaptive_lead_steps,
                "adaptive_lead_minutes": (
                    adaptive_lead_steps * step_minutes
                    if not np.isnan(adaptive_lead_steps)
                    else np.nan
                ),
                "adaptive_leads": bool(adaptive_leads),
                "max_evidence_pre_cons": max_evidence_pre_cons,
                "mean_evidence_pre_cons": mean_evidence_pre_cons,
                "elevated_evidence_threshold": float(elevated_evidence_threshold),
                "elevated_evidence_flag": elevated_evidence_flag,
                "elevated_evidence_count_pre_cons": elevated_evidence_count,
                "mean_adaptive_threshold_pre_cons": mean_adaptive_threshold_pre_cons,
                "min_adaptive_threshold_pre_cons": min_adaptive_threshold_pre_cons,
                "inertial_candidate": bool(adaptive_leads and elevated_evidence_flag),
            }
        )

    return pd.DataFrame(out_rows)


def summarize_response_inertia(
    window_df: pd.DataFrame,
    clip_evidence_upper: float = 20.0,
) -> pd.DataFrame:
    """
    Build a compact summary table from the per-window response-inertia table.

    Evidence summaries are clipped from above for reporting only, to avoid
    overly large descriptive values caused by near-zero MAD in very flat series.
    """
    if len(window_df) == 0:
        return pd.DataFrame(
            [
                {
                    "n_windows": 0,
                    "adaptive_lead_windows": 0,
                    "adaptive_lead_rate": np.nan,
                    "mean_adaptive_lead_steps": np.nan,
                    "mean_adaptive_lead_minutes": np.nan,
                    "lead_windows_with_elevated_evidence": 0,
                    "lead_windows_with_elevated_evidence_rate": np.nan,
                    "mean_clipped_max_evidence_pre_cons_on_lead_windows": np.nan,
                    "mean_clipped_mean_evidence_pre_cons_on_lead_windows": np.nan,
                    "inertial_candidate_windows": 0,
                    "inertial_candidate_rate": np.nan,
                    "evidence_clip_upper": float(clip_evidence_upper),
                }
            ]
        )

    lead_df = window_df[window_df["adaptive_leads"]].copy()
    inertial_df = window_df[window_df["inertial_candidate"]].copy()

    if len(lead_df) > 0:
        clipped_max = lead_df["max_evidence_pre_cons"].clip(upper=clip_evidence_upper)
        clipped_mean = lead_df["mean_evidence_pre_cons"].clip(upper=clip_evidence_upper)

        mean_clipped_max = float(clipped_max.mean())
        mean_clipped_mean = float(clipped_mean.mean())
    else:
        mean_clipped_max = np.nan
        mean_clipped_mean = np.nan

    out = {
        "n_windows": int(len(window_df)),
        "adaptive_lead_windows": int(len(lead_df)),
        "adaptive_lead_rate": float(len(lead_df) / len(window_df)),
        "mean_adaptive_lead_steps": (
            float(lead_df["adaptive_lead_steps"].mean())
            if len(lead_df) > 0
            else np.nan
        ),
        "mean_adaptive_lead_minutes": (
            float(lead_df["adaptive_lead_minutes"].mean())
            if len(lead_df) > 0
            else np.nan
        ),
        "lead_windows_with_elevated_evidence": (
            int(lead_df["elevated_evidence_flag"].sum())
            if len(lead_df) > 0
            else 0
        ),
        "lead_windows_with_elevated_evidence_rate": (
            float(lead_df["elevated_evidence_flag"].mean())
            if len(lead_df) > 0
            else np.nan
        ),
        "mean_clipped_max_evidence_pre_cons_on_lead_windows": mean_clipped_max,
        "mean_clipped_mean_evidence_pre_cons_on_lead_windows": mean_clipped_mean,
        "inertial_candidate_windows": int(len(inertial_df)),
        "inertial_candidate_rate": float(len(inertial_df) / len(window_df)),
        "evidence_clip_upper": float(clip_evidence_upper),
    }

    return pd.DataFrame([out])


def _fisher_exact_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """
    Compute a two-sided Fisher exact test for the 2x2 table:

        [[a, b],
         [c, d]]

    Returns
    -------
    tuple
        (odds_ratio, p_value)
    """
    if scipy_fisher_exact is not None:
        odds_ratio, p_value = scipy_fisher_exact(
            [[a, b], [c, d]],
            alternative="two-sided",
        )
        return float(odds_ratio), float(p_value)

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    n = row1 + row2

    def comb(n_: int, k_: int) -> int:
        if k_ < 0 or k_ > n_:
            return 0
        return math.comb(n_, k_)

    def hypergeom_prob(x: int) -> float:
        return comb(col1, x) * comb(col2, row1 - x) / comb(n, row1)

    lo = max(0, row1 - col2)
    hi = min(row1, col1)

    p_obs = hypergeom_prob(a)
    p_two_sided = 0.0
    for x in range(lo, hi + 1):
        px = hypergeom_prob(x)
        if px <= p_obs + 1e-12:
            p_two_sided += px

    if a == 0 or b == 0 or c == 0 or d == 0:
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        odds_ratio = (a * d) / (b * c)

    return float(odds_ratio), float(min(max(p_two_sided, 0.0), 1.0))


def summarize_lead_vs_nonlead(
    window_df: pd.DataFrame,
    elevated_threshold: float = 3.35,
) -> pd.DataFrame:
    """
    Compare adaptive-lead windows against non-lead windows using a recomputed
    elevated-evidence indicator based on max_evidence_pre_cons.
    """
    if len(window_df) == 0:
        return pd.DataFrame(
            [
                {
                    "elevated_evidence_threshold": float(elevated_threshold),
                    "lead_windows": 0,
                    "lead_windows_elevated": 0,
                    "lead_windows_elevated_rate": np.nan,
                    "nonlead_windows": 0,
                    "nonlead_windows_elevated": 0,
                    "nonlead_windows_elevated_rate": np.nan,
                    "odds_ratio": np.nan,
                    "fisher_p_value": np.nan,
                }
            ]
        )

    temp = window_df.copy()

    # Make sure adaptive_leads is truly boolean
    temp["adaptive_leads"] = temp["adaptive_leads"].fillna(False).astype(bool)

    temp["elevated_recomputed"] = (
        pd.to_numeric(temp["max_evidence_pre_cons"], errors="coerce") >= elevated_threshold
    )

    lead_df = temp[temp["adaptive_leads"]].copy()
    nonlead_df = temp[~temp["adaptive_leads"]].copy()

    a = int(lead_df["elevated_recomputed"].sum())
    b = int(len(lead_df) - a)
    c = int(nonlead_df["elevated_recomputed"].sum())
    d = int(len(nonlead_df) - c)

    odds_ratio, fisher_p = _fisher_exact_2x2(a, b, c, d)

    return pd.DataFrame(
        [
            {
                "elevated_evidence_threshold": float(elevated_threshold),
                "lead_windows": int(len(lead_df)),
                "lead_windows_elevated": int(a),
                "lead_windows_elevated_rate": (
                    float(a / len(lead_df)) if len(lead_df) > 0 else np.nan
                ),
                "nonlead_windows": int(len(nonlead_df)),
                "nonlead_windows_elevated": int(c),
                "nonlead_windows_elevated_rate": (
                    float(c / len(nonlead_df)) if len(nonlead_df) > 0 else np.nan
                ),
                "odds_ratio": odds_ratio,
                "fisher_p_value": fisher_p,
            }
        ]
    )
def summarize_response_inertia_threshold_sensitivity(
    window_df: pd.DataFrame,
    thresholds: Sequence[float] = (3.0, 3.35, 3.5, 4.0),
) -> pd.DataFrame:
    """
    Recompute the lead-vs-nonlead comparison over multiple elevated-evidence
    thresholds.
    """
    rows: List[Dict[str, object]] = []

    for threshold in thresholds:
        comparison = summarize_lead_vs_nonlead(
            window_df,
            elevated_threshold=float(threshold),
        ).iloc[0].to_dict()
        rows.append(comparison)

    return pd.DataFrame(rows)


def build_response_inertia_diagnostic_tables(
    window_df: pd.DataFrame,
    thresholds: Sequence[float] = (3.0, 3.35, 3.5, 4.0),
    clip_evidence_upper: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper returning:
      1. overall summary,
      2. lead-vs-nonlead comparison at threshold 3.35,
      3. threshold-sensitivity summary.
    """
    overall_df = summarize_response_inertia(
        window_df=window_df,
        clip_evidence_upper=clip_evidence_upper,
    )

    lead_vs_nonlead_df = summarize_lead_vs_nonlead(
        window_df=window_df,
        elevated_threshold=3.35,
    )

    threshold_sensitivity_df = summarize_response_inertia_threshold_sensitivity(
        window_df=window_df,
        thresholds=thresholds,
    )

    return overall_df, lead_vs_nonlead_df, threshold_sensitivity_df