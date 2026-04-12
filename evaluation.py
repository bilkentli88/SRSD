import numpy as np
import pandas as pd


def infer_time_step(df, timestamp_col="timestamp"):
    ts = pd.to_datetime(df[timestamp_col])
    diffs = ts.diff().dropna()

    if len(diffs) == 0:
        raise ValueError("Cannot infer time step from fewer than 2 timestamps.")

    mode_vals = diffs.mode()
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]

    return diffs.median()


def build_label_windows(label_times, step, half_window_steps=24):
    windows = []
    delta = half_window_steps * step

    for t in pd.to_datetime(label_times):
        start = t - delta
        end = t + delta
        windows.append((t, start, end))

    return windows


def merge_overlapping_windows(simple_windows):
    if not simple_windows:
        return []

    simple_windows = sorted(simple_windows, key=lambda x: x[0])
    merged = [simple_windows[0]]

    for start, end in simple_windows[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def find_alarm_onsets(alarm_array):
    alarm_array = np.asarray(alarm_array, dtype=int)

    if len(alarm_array) == 0:
        return np.array([], dtype=int)

    onsets = []
    if alarm_array[0] == 1:
        onsets.append(0)

    for i in range(1, len(alarm_array)):
        if alarm_array[i - 1] == 0 and alarm_array[i] == 1:
            onsets.append(i)

    return np.array(onsets, dtype=int)


def onset_in_any_window(onset_time, merged_windows):
    for start, end in merged_windows:
        if start <= onset_time <= end:
            return True
    return False


def evaluate_policy(
    df,
    alarm_col,
    label_times,
    timestamp_col="timestamp",
    half_window_steps=24
):
    timestamps = pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
    alarm = np.asarray(df[alarm_col], dtype=int)

    step = infer_time_step(df, timestamp_col=timestamp_col)

    label_windows = build_label_windows(label_times, step, half_window_steps=half_window_steps)
    merged_windows = merge_overlapping_windows([(start, end) for _, start, end in label_windows])

    onsets = find_alarm_onsets(alarm)
    onset_times = timestamps.iloc[onsets].reset_index(drop=True)

    hits = 0
    delays_steps = []

    for label_time, window_start, window_end in label_windows:
        in_window_mask = (timestamps >= window_start) & (timestamps <= window_end)
        in_window_indices = np.where(in_window_mask)[0]

        if len(in_window_indices) == 0:
            continue

        # Hit if alarm is active anywhere in the full window
        alarmed_indices = [idx for idx in in_window_indices if alarm[idx] == 1]

        if len(alarmed_indices) == 0:
            continue

        hits += 1

        # Delay logic:
        # 1) if already in alarm at the label time, delay = 0
        # 2) else find first alarmed point at or after the label time within the window
        label_or_after_indices = [idx for idx in in_window_indices if timestamps.iloc[idx] >= label_time]

        delay_recorded = False

        # Case 1: already active at label time
        active_at_label = False
        for idx in label_or_after_indices:
            if timestamps.iloc[idx] == label_time and alarm[idx] == 1:
                active_at_label = True
                break

        if active_at_label:
            delays_steps.append(0.0)
            delay_recorded = True
        else:
            # Also treat "already active before label and still active after label" as zero delay
            prev_indices = [idx for idx in in_window_indices if timestamps.iloc[idx] < label_time]
            if len(prev_indices) > 0 and len(label_or_after_indices) > 0:
                last_before = prev_indices[-1]
                first_after_candidates = [idx for idx in label_or_after_indices]
                if len(first_after_candidates) > 0:
                    first_after = first_after_candidates[0]
                    if alarm[last_before] == 1 and alarm[first_after] == 1:
                        delays_steps.append(0.0)
                        delay_recorded = True

        # Case 2: first alarm after label time
        if not delay_recorded:
            after_alarm_indices = [idx for idx in label_or_after_indices if alarm[idx] == 1]
            if len(after_alarm_indices) > 0:
                first_idx = after_alarm_indices[0]
                delay = (timestamps.iloc[first_idx] - label_time) / step
                delays_steps.append(float(delay))
                delay_recorded = True

        # Fallback safety: if somehow only pre-label alarm existed, count as zero-delay hit
        if not delay_recorded:
            delays_steps.append(0.0)

    misses = len(label_times) - hits

    false_alarm_onsets = 0
    for t in onset_times:
        if not onset_in_any_window(t, merged_windows):
            false_alarm_onsets += 1

    if len(delays_steps) > 0:
        mean_delay_steps = float(np.mean(delays_steps))
        median_delay_steps = float(np.median(delays_steps))
        mean_delay_minutes = float(mean_delay_steps * (step / pd.Timedelta(minutes=1)))
    else:
        mean_delay_steps = np.nan
        median_delay_steps = np.nan
        mean_delay_minutes = np.nan

    return {
        "alarm_col": alarm_col,
        "n_labels": int(len(label_times)),
        "hits": int(hits),
        "misses": int(misses),
        "hit_rate": float(hits / len(label_times)) if len(label_times) > 0 else np.nan,
        "mean_delay_steps": mean_delay_steps,
        "median_delay_steps": median_delay_steps,
        "mean_delay_minutes": mean_delay_minutes,
        "false_alarm_onsets": int(false_alarm_onsets),
        "total_alarm_onsets": int(len(onsets)),
        "time_step_minutes": float(step / pd.Timedelta(minutes=1)),
        "half_window_steps": int(half_window_steps),
    }


def evaluate_all_policies(
    df,
    label_times,
    policy_alarm_cols,
    timestamp_col="timestamp",
    half_window_steps=24
):
    rows = []

    for col in policy_alarm_cols:
        result = evaluate_policy(
            df=df,
            alarm_col=col,
            label_times=label_times,
            timestamp_col=timestamp_col,
            half_window_steps=half_window_steps
        )
        rows.append(result)

    return pd.DataFrame(rows)