"""Identify P1–P10 golf swing positions from smoothed pose results."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from analysis.pose_runner import PoseResult


_LABELS = {
    1:  "P1: Address",
    2:  "P2: Takeaway",
    3:  "P3: Backswing",
    4:  "P4: Top",
    5:  "P5: Downswing",
    6:  "P6: Pre-Impact",
    7:  "P7: Impact",
    8:  "P8: Follow-Through",
    9:  "P9: Extension",
    10: "P10: Finish",
}

_SHORT = {
    1:  "Address",
    2:  "Takeaway",
    3:  "Backswing",
    4:  "Top",
    5:  "Downswing",
    6:  "Pre-Impact",
    7:  "Impact",
    8:  "Follow-Through",
    9:  "Extension",
    10: "Finish",
}


@dataclass
class SwingPhases:
    """Frame indices for the 10 standard golf swing positions (P1–P10).

    Coordinate convention (face-on / cam0 view):
      Y=0 is the top of the frame, Y=1 the bottom.
      A rising wrist means Y is *decreasing*.

    P1  – Address             (setup, wrist low, stationary)
    P2  – Club parallel       (takeaway, wrist ≈ hip height)
    P3  – Arm parallel        (backswing, wrist ≈ shoulder height)
    P4  – Top of backswing    (wrist highest point)
    P5  – Arm parallel        (downswing, wrist ≈ shoulder height)
    P6  – Club parallel       (downswing, wrist ≈ hip height)
    P7  – Impact
    P8  – Club parallel       (follow-through, wrist ≈ hip height)
    P9  – Arm parallel        (follow-through, wrist ≈ shoulder height)
    P10 – Finish              (wrist highest in follow-through)
    """
    p1:  int
    p2:  int
    p3:  int
    p4:  int
    p5:  int
    p6:  int
    p7:  int
    p8:  int
    p9:  int
    p10: int
    total_frames: int

    # ── Backward-compatibility aliases (used by saver, old metrics code) ───────
    @property
    def address_idx(self) -> int:       return self.p1
    @property
    def backswing_top_idx(self) -> int: return self.p4
    @property
    def impact_idx(self) -> int:        return self.p7
    @property
    def follow_through_idx(self) -> int: return self.p9

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def p_frame(self, n: int) -> int:
        """Return the frame index for position Pn (1-based, 1–10)."""
        return [
            self.p1, self.p2, self.p3, self.p4, self.p5,
            self.p6, self.p7, self.p8, self.p9, self.p10,
        ][n - 1]

    def p_label(self, n: int) -> str:
        """Full label string, e.g. 'P4: Top'."""
        return _LABELS.get(n, f"P{n}")

    def phase_for_frame(self, idx: int) -> str:
        """Short phase name for on-screen display at a given frame index."""
        pairs = [
            (self.p1,  _SHORT[1]),
            (self.p2,  _SHORT[2]),
            (self.p3,  _SHORT[3]),
            (self.p4,  _SHORT[4]),
            (self.p5,  _SHORT[5]),
            (self.p6,  _SHORT[6]),
            (self.p7,  _SHORT[7]),
            (self.p8,  _SHORT[8]),
            (self.p9,  _SHORT[9]),
            (self.p10, _SHORT[10]),
        ]
        label = _SHORT[1]
        for frame_idx, name in pairs:
            if idx >= frame_idx:
                label = name
            else:
                break
        return label


def detect_phases(
    results: list[PoseResult],
    impact_frame_idx: int = 0,
) -> SwingPhases:
    """Detect P1–P10 swing positions from face-on (cam0) pose results.

    The algorithm tracks the RIGHT_WRIST Y-position relative to the shoulder
    and hip reference heights measured at address.

    Args:
        results: Smoothed PoseResult list from the face-on camera.
        impact_frame_idx: Frame closest to audio trigger / ball contact,
            used to anchor P7.  If 0 or ≤ estimated backswing top, P7 is
            estimated from peak downward wrist velocity instead.
    """
    n = len(results)
    if n == 0:
        return SwingPhases(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # ── Extract wrist / shoulder / hip Y tracks ────────────────────────────────
    wrist_y:    list = []
    shoulder_y: list = []
    hip_y:      list = []

    for r in results:
        rw = r.lm("RIGHT_WRIST")
        rs = r.lm("RIGHT_SHOULDER")
        ls = r.lm("LEFT_SHOULDER")
        rh = r.lm("RIGHT_HIP")
        lh = r.lm("LEFT_HIP")

        wrist_y.append(rw.y if rw else None)

        if rs and ls:
            shoulder_y.append((rs.y + ls.y) / 2)
        elif rs:
            shoulder_y.append(rs.y)
        elif ls:
            shoulder_y.append(ls.y)
        else:
            shoulder_y.append(None)

        if rh and lh:
            hip_y.append((rh.y + lh.y) / 2)
        elif rh:
            hip_y.append(rh.y)
        elif lh:
            hip_y.append(lh.y)
        else:
            hip_y.append(None)

    wrist_y    = _fill_none(wrist_y)
    shoulder_y = _fill_none(shoulder_y)
    hip_y      = _fill_none(hip_y)

    arr_w = np.array(wrist_y,    dtype=float)
    arr_s = np.array(shoulder_y, dtype=float)
    arr_h = np.array(hip_y,      dtype=float)

    # Light additional smoothing for phase-boundary detection
    # (landmark data is already Gaussian-smoothed by PoseRunner)
    if n >= 5:
        k5 = np.ones(5) / 5
        arr_w = np.convolve(arr_w, k5, mode='same')

    velocity = np.diff(arr_w, prepend=arr_w[0])

    # ── P1: Address — last stable frame before meaningful wrist movement ────────
    p1 = 0
    for i in range(len(velocity)):
        if abs(velocity[i]) > 0.003:
            p1 = max(0, i - 2)
            break

    # ── P4: Top of backswing — wrist highest (minimum Y) ───────────────────────
    search_end = min(n - 1, int(n * 0.70))
    p4 = p1 + int(np.argmin(arr_w[p1:search_end + 1]))

    # ── P7: Impact ──────────────────────────────────────────────────────────────
    if impact_frame_idx > p4:
        p7 = min(impact_frame_idx, n - 1)
    else:
        # Estimate: peak positive (downward) velocity after the backswing top
        if p4 < n - 1:
            p7 = p4 + int(np.argmax(velocity[p4:]))
            p7 = min(p7, n - 1)
        else:
            p7 = p4

    # ── P10: Finish — wrist highest point in follow-through ────────────────────
    if p7 < n - 1:
        p10 = p7 + int(np.argmin(arr_w[p7:]))
        p10 = min(p10, n - 1)
    else:
        p10 = n - 1

    # Reference heights measured at P1
    ref_s = float(arr_s[p1])   # mid-shoulder Y
    ref_h = float(arr_h[p1])   # mid-hip Y

    # ── Intermediate positions — frame closest to the reference height ──────────
    # Backswing: wrist rises from hip level → shoulder level → top
    p3 = _find_crossing(arr_w, ref_s, p1, p4)   # arm parallel
    p2 = _find_crossing(arr_w, ref_h, p1, p3)   # club parallel (takeaway)

    # Downswing: wrist falls from top → shoulder level → hip level → impact
    p5 = _find_crossing(arr_w, ref_s, p4, p7)   # arm parallel
    p6 = _find_crossing(arr_w, ref_h, p5, p7)   # club parallel

    # Follow-through: wrist rises from impact → hip level → shoulder level → finish
    p8 = _find_crossing(arr_w, ref_h, p7, p10)  # club parallel
    p9 = _find_crossing(arr_w, ref_s, p8, p10)  # arm parallel

    return SwingPhases(
        p1=p1, p2=p2, p3=p3, p4=p4,
        p5=p5, p6=p6, p7=p7, p8=p8,
        p9=p9, p10=p10,
        total_frames=n,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_crossing(arr: np.ndarray, target: float, start: int, end: int) -> int:
    """Return the frame in [start, end] where arr is closest to target."""
    if end <= start:
        return start
    return start + int(np.argmin(np.abs(arr[start:end + 1] - target)))


def _fill_none(values: list) -> list:
    """Linear-interpolation gap fill for a list that may contain None values."""
    result = list(values)
    n = len(result)
    first = next((v for v in result if v is not None), 0.5)
    i = 0
    while i < n:
        if result[i] is None:
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            left  = result[i - 1] if i > 0 else (result[j] if j < n else first)
            right = result[j] if j < n else left
            span  = j - i + 1
            for k in range(i, j):
                t = (k - i + 1) / span
                result[k] = left + t * (right - left)
            i = j
        else:
            first = result[i]
            i += 1
    return result
