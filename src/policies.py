import numpy as np


def run_aggressive_policy(df, evidence_col="evidence", threshold=3.0):
    out = df.copy()
    evidence = out[evidence_col].values

    alarm = np.zeros(len(out), dtype=int)

    for t in range(len(out)):
        e = evidence[t]

        if np.isnan(e):
            alarm[t] = 0
        elif e >= threshold:
            alarm[t] = 1
        else:
            alarm[t] = 0

    out["alarm_aggressive"] = alarm
    return out


def run_conservative_policy(
    df,
    evidence_col="evidence",
    threshold=4.0,
    enter_count=3,
    exit_count=3
):
    out = df.copy()
    evidence = out[evidence_col].values

    alarm = np.zeros(len(out), dtype=int)

    state = 0
    high_streak = 0
    low_streak = 0

    for t in range(len(out)):
        e = evidence[t]

        if np.isnan(e):
            alarm[t] = state
            continue

        if state == 0:
            if e >= threshold:
                high_streak += 1
            else:
                high_streak = 0

            if high_streak >= enter_count:
                state = 1
                low_streak = 0

        else:
            if e < threshold:
                low_streak += 1
            else:
                low_streak = 0

            if low_streak >= exit_count:
                state = 0
                high_streak = 0

        alarm[t] = state

    out["alarm_conservative"] = alarm
    return out


def run_adaptive_policy(
    df,
    evidence_col="evidence",
    base_threshold=3.35,
    enter_margin=0.0,
    exit_margin=0.45,
    adaptation_rate=0.065,
    relaxation_rate=0.04,
    switch_penalty=0.27,
    min_threshold=2.45,
    max_threshold=4.7
):
    out = df.copy()
    evidence = out[evidence_col].values

    alarm = np.zeros(len(out), dtype=int)
    threshold_series = np.full(len(out), np.nan, dtype=float)

    state = 0
    theta = base_threshold
    pressure = 0.0
    switch_memory = 0.0

    for t in range(len(out)):
        e = evidence[t]

        if np.isnan(e):
            alarm[t] = 0
            threshold_series[t] = theta
            continue

        gap = e - theta
        pressure = 0.85 * pressure + gap
        switch_memory = 0.85 * switch_memory

        theta = (
            theta
            - adaptation_rate * max(pressure, 0.0)
            + switch_penalty * switch_memory
            + relaxation_rate * (base_threshold - theta)
        )

        theta = max(min_threshold, min(max_threshold, theta))

        enter_threshold = theta + enter_margin
        exit_threshold = max(min_threshold, theta - exit_margin)

        previous_state = state

        if state == 0:
            if e >= enter_threshold:
                state = 1
        else:
            if e < exit_threshold:
                state = 0

        if state != previous_state:
            switch_memory += 1.0

        alarm[t] = state
        threshold_series[t] = theta

    out["alarm_adaptive"] = alarm
    out["adaptive_threshold"] = threshold_series
    return out