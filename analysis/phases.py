"""Detect swing phase boundaries from pose results."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from analysis.pose_runner import PoseResult


@dataclass
class SwingPhases:
    address_idx: int
    backswing_top_idx: int
    impact_idx: int
    follow_through_idx: int
    total_frames: int

    def phase_for_frame(self, idx: int) -> str:
        if idx <= self.address_idx:
            return "Address"
        if idx <= self.backswing_top_idx:
            return "Backswing"
        if idx <= self.impact_idx:
            return "Downswing"
        if idx <= self.follow_through_idx:
            return "Follow-Through"
        return "Finish"


def detect_phases(results: list[PoseResult]) -> SwingPhases:
    """Detect swing phases by tracking RIGHT_WRIST vertical motion (cam0)."""
    n = len(results)
    if n == 0:
        return SwingPhases(0, 0, 0, 0, 0)

    # Extract right-wrist Y positions (normalised; 0=top, 1=bottom)
    wrist_y = []
    for r in results:
        lm = r.lm("RIGHT_WRIST")
        wrist_y.append(lm.y if lm else None)

    # Fill None gaps with linear interpolation
    wrist_y = _fill_none(wrist_y)

    arr = np.array(wrist_y, dtype=float)

    # Smooth with a 7-frame rolling average
    kernel = np.ones(7) / 7
    if len(arr) >= 7:
        arr_smooth = np.convolve(arr, kernel, mode="same")
    else:
        arr_smooth = arr.copy()

    # Velocity (positive = wrist moving down, i.e. downswing in image coords)
    velocity = np.diff(arr_smooth, prepend=arr_smooth[0])

    # ── Address: last frame before wrist starts moving meaningfully ────────────
    address_idx = 0
    movement_threshold = 0.003
    for i in range(len(velocity)):
        if abs(velocity[i]) > movement_threshold:
            address_idx = max(0, i - 2)
            break

    # ── Backswing top: wrist reaches highest point (minimum Y) ─────────────────
    # Search from address to ~60% of the clip
    search_end = min(n - 1, int(n * 0.70))
    backswing_top_idx = address_idx
    if search_end > address_idx:
        segment = arr_smooth[address_idx:search_end]
        backswing_top_idx = address_idx + int(np.argmin(segment))

    # ── Impact: maximum downward velocity after backswing top ──────────────────
    impact_idx = backswing_top_idx
    if backswing_top_idx < n - 1:
        downswing_vel = velocity[backswing_top_idx:]
        # Find peak positive velocity (fastest downward = impact)
        impact_idx = backswing_top_idx + int(np.argmax(downswing_vel))

    impact_idx = min(impact_idx, n - 1)

    # ── Follow-through: 20 frames after impact (or end of clip) ────────────────
    follow_through_idx = min(impact_idx + 20, n - 1)

    return SwingPhases(
        address_idx=address_idx,
        backswing_top_idx=backswing_top_idx,
        impact_idx=impact_idx,
        follow_through_idx=follow_through_idx,
        total_frames=n,
    )


def _fill_none(values: list) -> list:
    """Replace None entries with linear interpolation from neighbours."""
    result = list(values)
    n = len(result)
    # Find first non-None to use as left fill
    first_val = next((v for v in result if v is not None), 0.5)
    last_val = first_val

    i = 0
    while i < n:
        if result[i] is None:
            # Find next non-None
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            right_val = result[j] if j < n else last_val
            left_val = result[i - 1] if i > 0 else right_val
            # Interpolate
            span = j - i + 1
            for k in range(i, j):
                t = (k - i + 1) / span
                result[k] = left_val + t * (right_val - left_val)
            i = j
        else:
            last_val = result[i]
            i += 1
    return result
