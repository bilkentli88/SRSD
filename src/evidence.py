import numpy as np
import pandas as pd


def compute_online_robust_evidence(
    df,
    value_col="value",
    window=48,
    min_history=24,
    eps=1e-8
):
    """
    Computes a causal anomaly evidence score using:
      - rolling median of past values
      - rolling MAD of past values
      - evidence_t = |x_t - median_past| / (MAD_past + eps)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns [timestamp, value_col].
    value_col : str
        Name of the numeric value column.
    window : int
        Number of past observations used for rolling statistics.
    min_history : int
        Minimum number of past observations required before scoring starts.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    out : pandas.DataFrame
        Copy of df with added columns:
          - baseline
          - mad
          - evidence
    """
    out = df.copy().reset_index(drop=True)

    x = out[value_col].astype(float).values
    n = len(x)

    baseline = np.full(n, np.nan, dtype=float)
    mad_vals = np.full(n, np.nan, dtype=float)
    evidence = np.full(n, np.nan, dtype=float)

    for t in range(n):
        start = max(0, t - window)
        history = x[start:t]   # past only, excludes current point

        if len(history) < min_history:
            continue

        med = np.median(history)
        mad = np.median(np.abs(history - med))

        baseline[t] = med
        mad_vals[t] = mad
        evidence[t] = abs(x[t] - med) / (mad + eps)

    out["baseline"] = baseline
    out["mad"] = mad_vals
    out["evidence"] = evidence

    return out