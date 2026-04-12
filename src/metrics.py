"""
Utility metrics for alarm sequences.
"""

from __future__ import annotations

import numpy as np


def count_switches(alarm_array: np.ndarray) -> int:
    """
    Count the number of state changes in a binary alarm sequence.

    Parameters
    ----------
    alarm_array : np.ndarray
        Sequence of binary alarm states.

    Returns
    -------
    int
        Number of transitions between consecutive time points.
    """
    alarm = np.asarray(alarm_array, dtype=int)

    if len(alarm) < 2:
        return 0

    return int(np.sum(alarm[1:] != alarm[:-1]))