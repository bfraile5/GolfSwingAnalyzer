"""Swing metrics and comprehensive P1–P10 position analysis. Pure functions, no I/O."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from analysis.pose_runner import PoseResult
from analysis.phases import SwingPhases

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ── SwingMetrics (existing, unchanged interface) ──────────────────────────────

@dataclass
class SwingMetrics:
    # Scores 0–100
    spine_angle_score: float = 0.0
    hip_rotation_score: float = 0.0
    knee_flex_score: float = 0.0
    head_stability_score: float = 0.0
    arm_extension_score: float = 0.0
    swing_plane_score: float = 0.0
    overall_score: float = 0.0

    # Raw values for display
    spine_angle_deg: float = 0.0
    hip_rotation_deg: float = 0.0
    knee_flex_deg: float = 0.0
    head_stddev: float = 0.0
    arm_extension_deg: float = 0.0

    tips: dict = field(default_factory=dict)


def compute_metrics(
    results_cam0: list[PoseResult],
    results_cam2: list[PoseResult],
    phases: SwingPhases,
) -> SwingMetrics:
    m = SwingMetrics()

    m.spine_angle_deg, m.spine_angle_score = _spine_angle(results_cam0, phases.p1)
    m.tips["Spine Tilt"] = _spine_tip(m.spine_angle_deg)

    m.hip_rotation_deg, m.hip_rotation_score = _hip_rotation(
        results_cam0, phases.p1, phases.p7)
    m.tips["Hip Rotation"] = _hip_tip(m.hip_rotation_deg)

    m.knee_flex_deg, m.knee_flex_score = _knee_flex(results_cam0, phases.p1)
    m.tips["Knee Flex"] = _knee_tip(m.knee_flex_deg)

    m.head_stddev, m.head_stability_score = _head_stability(
        results_cam0, phases.p1, phases.p7)
    m.tips["Head Stability"] = _head_tip(m.head_stddev)

    m.arm_extension_deg, m.arm_extension_score = _arm_extension(
        results_cam2, phases.p7)
    m.tips["Arm Extension"] = _arm_tip(m.arm_extension_deg)

    plane_angle, m.swing_plane_score = _swing_plane(
        results_cam2, phases.p1, phases.p4)
    m.tips["Swing Plane"] = _plane_tip(plane_angle)

    scores = [
        m.spine_angle_score, m.hip_rotation_score, m.knee_flex_score,
        m.head_stability_score, m.arm_extension_score, m.swing_plane_score,
    ]
    m.overall_score = round(sum(scores) / len(scores), 1)
    return m


# ── P1–P10 Position Analysis ──────────────────────────────────────────────────

@dataclass
class PositionData:
    """Body angles and tracking data captured at one of the P1–P10 positions."""
    frame_idx: int
    label: str
    # Absolute body angles at this frame
    spine_angle: float = 0.0          # degrees forward tilt from vertical
    knee_flex: float = 0.0            # lead knee joint angle (degrees; ~160° = good flex)
    lead_arm_angle: float = 0.0       # lead shoulder-elbow-wrist angle (degrees)
    wrist_height_ratio: float = 0.0   # (hip_y - wrist_y)/(hip_y - shoulder_y); >1 = above shoulder
    head_x: float = 0.0               # nose X (normalised image coords)
    head_y: float = 0.0               # nose Y (normalised image coords)
    # Rotation deltas relative to P1 (degrees, always positive)
    hip_rotation_delta: float = 0.0
    shoulder_rotation_delta: float = 0.0
    # Euclidean head drift from P1 nose position (normalised coords)
    head_drift: float = 0.0


@dataclass
class SwingAnalysis:
    """Full P1–P10 swing analysis: per-position body data and text evaluations."""
    positions: list[PositionData] = field(default_factory=list)  # indices 0–9 → P1–P10

    # Text evaluation per swing aspect
    address_evaluation: str = ""
    backswing_evaluation: str = ""
    transition_evaluation: str = ""
    impact_evaluation: str = ""
    follow_through_evaluation: str = ""

    # Summary statistics
    max_shoulder_rotation: float = 0.0
    max_hip_rotation: float = 0.0


def compute_swing_analysis(
    results_cam0: list[PoseResult],
    results_cam2: list[PoseResult],
    phases: SwingPhases,
) -> SwingAnalysis:
    """Compute detailed body data and text evaluation for all P1–P10 positions."""
    if not results_cam0:
        return SwingAnalysis()

    # Reference values at P1 (address)
    p1_hip_yaw = _yaw_angle(results_cam0, phases.p1, "LEFT_HIP",      "RIGHT_HIP")
    p1_sho_yaw = _yaw_angle(results_cam0, phases.p1, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    p1_head    = _nose_xy(results_cam0, phases.p1)

    positions: list[PositionData] = []
    for n in range(1, 11):
        fidx  = phases.p_frame(n)
        label = phases.p_label(n)

        spine_ang, _ = _spine_angle(results_cam0, fidx)
        knee_deg, _  = _knee_flex(results_cam0, fidx)
        lead_arm     = _lead_arm_angle(results_cam0, fidx)
        wrist_ratio  = _wrist_height_ratio(results_cam0, fidx)
        head_xy      = _nose_xy(results_cam0, fidx)

        hip_yaw = _yaw_angle(results_cam0, fidx, "LEFT_HIP",      "RIGHT_HIP")
        sho_yaw = _yaw_angle(results_cam0, fidx, "LEFT_SHOULDER", "RIGHT_SHOULDER")

        hip_delta = _angle_delta(hip_yaw, p1_hip_yaw)
        sho_delta = _angle_delta(sho_yaw, p1_sho_yaw)

        hx, hy = head_xy if head_xy else (0.0, 0.0)
        drift  = math.hypot(hx - p1_head[0], hy - p1_head[1]) if p1_head else 0.0

        positions.append(PositionData(
            frame_idx=fidx,
            label=label,
            spine_angle=spine_ang,
            knee_flex=knee_deg,
            lead_arm_angle=lead_arm,
            wrist_height_ratio=wrist_ratio,
            head_x=hx,
            head_y=hy,
            hip_rotation_delta=hip_delta,
            shoulder_rotation_delta=sho_delta,
            head_drift=drift,
        ))

    p1d, p2d, p3d, p4d, p5d = positions[0], positions[1], positions[2], positions[3], positions[4]
    p7d, p8d, p9d, p10d     = positions[6], positions[7], positions[8], positions[9]

    return SwingAnalysis(
        positions=positions,
        address_evaluation     = _eval_address(p1d),
        backswing_evaluation   = _eval_backswing(p2d, p3d, p4d),
        transition_evaluation  = _eval_transition(p4d, p5d),
        impact_evaluation      = _eval_impact(p7d),
        follow_through_evaluation = _eval_follow_through(p8d, p9d, p10d),
        max_shoulder_rotation  = max(p.shoulder_rotation_delta for p in positions),
        max_hip_rotation       = max(p.hip_rotation_delta       for p in positions),
    )


# ── Text evaluation helpers ────────────────────────────────────────────────────

def _eval_address(p: PositionData) -> str:
    notes = []
    if p.spine_angle < 25:
        notes.append("Stand taller at address — not enough forward spine tilt")
    elif p.spine_angle > 50:
        notes.append("Reduce spine tilt — too bent over the ball")
    else:
        notes.append(f"Good spine tilt at address ({p.spine_angle:.0f}\u00b0)")

    if p.knee_flex < 145:
        notes.append("Too much knee bend — use a light athletic flex, not a squat")
    elif p.knee_flex > 170:
        notes.append("Flex your knees more at address for a more athletic setup")
    else:
        notes.append("Good knee flex at address")

    if p.lead_arm_angle < 150:
        notes.append("Lead arm bent at address — straighten it to set a wider arc")
    return "  |  ".join(notes)


def _eval_backswing(p2: PositionData, p3: PositionData, p4: PositionData) -> str:
    notes = []
    # Shoulder turn at top
    if p4.shoulder_rotation_delta < 70:
        notes.append(
            f"Restricted shoulder turn at top ({p4.shoulder_rotation_delta:.0f}\u00b0) — aim for 90\u00b0")
    elif p4.shoulder_rotation_delta > 110:
        notes.append(f"Over-rotation at the top ({p4.shoulder_rotation_delta:.0f}\u00b0)")
    else:
        notes.append(f"Good shoulder rotation to the top ({p4.shoulder_rotation_delta:.0f}\u00b0)")

    # X-factor (shoulder rotation − hip rotation at top = power coil)
    x_factor = p4.shoulder_rotation_delta - p4.hip_rotation_delta
    if x_factor < 20:
        notes.append("Hips over-rotating on backswing — restrict them to build more power coil")
    elif x_factor > 55:
        notes.append(f"Excellent X-factor separation at top ({x_factor:.0f}\u00b0)")
    else:
        notes.append(f"Solid hip-shoulder separation at top (X-factor {x_factor:.0f}\u00b0)")
    return "  |  ".join(notes)


def _eval_transition(p4: PositionData, p5: PositionData) -> str:
    """Evaluate transition quality: hips should lead upper body in the downswing."""
    # At P5 (arm parallel, downswing) hips should have unwound more than shoulders
    # relative to their P4 (top) values — i.e. hips fired first.
    hip_unwind = p4.hip_rotation_delta - p5.hip_rotation_delta
    sho_unwind = p4.shoulder_rotation_delta - p5.shoulder_rotation_delta
    lead = hip_unwind - sho_unwind
    if lead > 8:
        return (f"Good hip lead in the transition — "
                f"hips unwind {hip_unwind:.0f}\u00b0 vs shoulders {sho_unwind:.0f}\u00b0 through P5")
    elif lead > 0:
        return ("Marginal hip lead — try firing your lower body more aggressively "
                "to start the downswing")
    else:
        return ("Upper body is leading the downswing — "
                "initiate the transition by turning the hips before the arms come down")


def _eval_impact(p: PositionData) -> str:
    notes = []
    if p.hip_rotation_delta < 30:
        notes.append(
            f"Hips not open enough at impact ({p.hip_rotation_delta:.0f}\u00b0) — "
            "drive through with your lower body")
    elif p.hip_rotation_delta > 65:
        notes.append(f"Hips over-cleared at impact ({p.hip_rotation_delta:.0f}\u00b0)")
    else:
        notes.append(f"Good hip position at impact ({p.hip_rotation_delta:.0f}\u00b0 open)")

    if p.head_drift > 0.07:
        notes.append(
            f"Head moved significantly through impact (drift {p.head_drift:.3f}) — stay centered")
    else:
        notes.append("Good head stability through impact")

    if p.lead_arm_angle < 140:
        notes.append(
            "Lead arm collapsing at impact — keep a straight lead arm through the ball")
    else:
        notes.append("Good lead arm extension at impact")
    return "  |  ".join(notes)


def _eval_follow_through(p8: PositionData, p9: PositionData, p10: PositionData) -> str:
    notes = []
    if p10.shoulder_rotation_delta < 100:
        notes.append(
            f"Restricted finish ({p10.shoulder_rotation_delta:.0f}\u00b0) — "
            "commit to a full follow-through")
    else:
        notes.append(
            f"Full finish achieved ({p10.shoulder_rotation_delta:.0f}\u00b0 shoulder rotation)")

    if p10.head_drift > 0.10:
        notes.append("Head drifting noticeably in the follow-through — maintain your spine angle")
    return "  |  ".join(notes)


# ── Per-position body angle helpers ───────────────────────────────────────────

def _yaw_angle(
    results: list[PoseResult], idx: int, lm_a: str, lm_b: str
) -> float:
    """Horizontal yaw angle of the line from world landmark A to B (degrees)."""
    r = _safe_get(results, idx)
    if r is None:
        return 0.0
    a = r.wlm(lm_a)
    b = r.wlm(lm_b)
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(b.z - a.z, b.x - a.x))


def _angle_delta(a2: float, a1: float) -> float:
    """Absolute angular difference, wrapped to [0, 180]."""
    d = abs(a2 - a1)
    return d if d <= 180 else 360.0 - d


def _lead_arm_angle(results: list[PoseResult], idx: int) -> float:
    """Lead arm (left side) shoulder-elbow-wrist joint angle in degrees."""
    r = _safe_get(results, idx)
    if r is None:
        return 0.0
    s = r.wlm("LEFT_SHOULDER")
    e = r.wlm("LEFT_ELBOW")
    w = r.wlm("LEFT_WRIST")
    if None in (s, e, w):
        return 0.0
    return _joint_angle(s, e, w)


def _wrist_height_ratio(results: list[PoseResult], idx: int) -> float:
    """Normalised wrist height: 0 = at hip level, 1 = at shoulder level, >1 = above shoulder."""
    r = _safe_get(results, idx)
    if r is None:
        return 0.0
    wrist   = r.lm("RIGHT_WRIST")
    shoulder = r.lm("RIGHT_SHOULDER")
    hip     = r.lm("RIGHT_HIP")
    if None in (wrist, shoulder, hip):
        return 0.0
    span = hip.y - shoulder.y   # positive (hip is lower = larger Y in image coords)
    if abs(span) < 1e-4:
        return 0.0
    return (hip.y - wrist.y) / span


def _nose_xy(results: list[PoseResult], idx: int):
    """Return (x, y) of nose landmark, or None."""
    r = _safe_get(results, idx)
    if r is None:
        return None
    nose = r.lm("NOSE")
    return (nose.x, nose.y) if nose else None


# ── Individual metric functions (unchanged from original) ─────────────────────

def _spine_angle(results, address_idx):
    r = _safe_get(results, address_idx)
    ls = r.wlm("LEFT_SHOULDER")  if r else None
    lh = r.wlm("LEFT_HIP")       if r else None
    rs = r.wlm("RIGHT_SHOULDER") if r else None
    rh = r.wlm("RIGHT_HIP")      if r else None
    if None in (ls, lh, rs, rh):
        return 0.0, 50.0
    sx = (ls.x + rs.x) / 2; sy = (ls.y + rs.y) / 2; sz = (ls.z + rs.z) / 2
    hx = (lh.x + rh.x) / 2; hy = (lh.y + rh.y) / 2; hz = (lh.z + rh.z) / 2
    dx = sx - hx; dy = sy - hy; dz = sz - hz
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-6:
        return 0.0, 50.0
    cos_theta = dy / length
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
    score = _tolerance_score(angle, config.SPINE_ANGLE_IDEAL, config.SPINE_ANGLE_TOLERANCE)
    return round(angle, 1), score


def _hip_rotation(results, address_idx, impact_idx):
    def hip_angle(r):
        if r is None:
            return None
        lh = r.wlm("LEFT_HIP"); rh = r.wlm("RIGHT_HIP")
        if lh is None or rh is None:
            return None
        return math.degrees(math.atan2(rh.z - lh.z, rh.x - lh.x))
    r_addr = _safe_get(results, address_idx)
    r_imp  = _safe_get(results, impact_idx)
    a1 = hip_angle(r_addr); a2 = hip_angle(r_imp)
    if a1 is None or a2 is None:
        return 0.0, 50.0
    rotation = abs(a2 - a1)
    if rotation > 180:
        rotation = 360 - rotation
    score = min(100.0, (rotation / config.HIP_ROTATION_IDEAL) * 100)
    return round(rotation, 1), round(score, 1)


def _knee_flex(results, address_idx):
    r = _safe_get(results, address_idx)
    if r is None:
        return 0.0, 50.0
    hip = r.wlm("LEFT_HIP"); knee = r.wlm("LEFT_KNEE"); ankle = r.wlm("LEFT_ANKLE")
    if None in (hip, knee, ankle):
        hip = r.wlm("RIGHT_HIP"); knee = r.wlm("RIGHT_KNEE"); ankle = r.wlm("RIGHT_ANKLE")
    if None in (hip, knee, ankle):
        return 0.0, 50.0
    angle     = _joint_angle(hip, knee, ankle)
    ideal_mid = (config.KNEE_FLEX_IDEAL_MIN + config.KNEE_FLEX_IDEAL_MAX) / 2
    tolerance = (config.KNEE_FLEX_IDEAL_MAX - config.KNEE_FLEX_IDEAL_MIN) / 2
    score     = _tolerance_score(angle, ideal_mid, tolerance)
    return round(angle, 1), score


def _head_stability(results, address_idx, impact_idx):
    xs, ys = [], []
    for r in results[address_idx:impact_idx + 1]:
        nose = r.lm("NOSE")
        if nose:
            xs.append(nose.x); ys.append(nose.y)
    if len(xs) < 2:
        return 0.0, 50.0
    stddev = float(np.std(xs) + np.std(ys))
    score  = max(0.0, 100 * (1 - stddev / config.HEAD_STABILITY_MAX_STDDEV))
    return round(stddev, 4), round(score, 1)


def _arm_extension(results, impact_idx):
    r = _safe_get(results, impact_idx)
    if r is None:
        return 0.0, 50.0
    s = r.wlm("LEFT_SHOULDER"); e = r.wlm("LEFT_ELBOW"); w = r.wlm("LEFT_WRIST")
    if None in (s, e, w):
        s = r.wlm("RIGHT_SHOULDER"); e = r.wlm("RIGHT_ELBOW"); w = r.wlm("RIGHT_WRIST")
    if None in (s, e, w):
        return 0.0, 50.0
    angle = _joint_angle(s, e, w)
    score = _tolerance_score(angle, 170.0, 15.0)
    return round(angle, 1), score


def _swing_plane(results, address_idx, backswing_top_idx):
    pts = []
    for r in results[address_idx:min(backswing_top_idx + 1, len(results))]:
        wrist = r.lm("RIGHT_WRIST")
        if wrist:
            pts.append([wrist.x, wrist.y])
    if len(pts) < 4:
        return 0.0, 50.0
    pts_arr = np.array(pts)
    centred = pts_arr - pts_arr.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    direction = vt[0]
    angle = abs(math.degrees(math.atan2(direction[1], direction[0])))
    score = _tolerance_score(angle, 50.0, 15.0)
    return round(angle, 1), score


# ── Tip text ──────────────────────────────────────────────────────────────────

def _spine_tip(angle):
    if angle < 20: return "Stand taller - more forward spine tilt needed"
    if angle > 50: return "Reduce spine tilt - you are bent over too much"
    if abs(angle - 35) <= 5: return "Excellent spine tilt at address"
    return "Good spine angle - minor adjustment needed"

def _hip_tip(deg):
    if deg < 30: return "Drive your hips harder through the ball"
    if deg < 45: return "Good rotation - push for a bit more hip turn"
    return "Excellent hip rotation through impact"

def _knee_tip(angle):
    if angle < 145: return "Straighten up slightly - too much knee bend"
    if angle > 170: return "Flex your knees more at address"
    return "Good knee flex at setup"

def _head_tip(stddev):
    if stddev > 0.06: return "Keep your head still - it is moving during the swing"
    if stddev > 0.03: return "Good head position - try to reduce lateral sway"
    return "Excellent head stability throughout the swing"

def _arm_tip(angle):
    if angle < 140: return "Extend your arms more through impact"
    if angle > 175: return "Good extension - maintain through follow-through"
    return "Good arm extension at impact"

def _plane_tip(angle):
    if angle < 35: return "Swing plane is too flat - steepen your arc"
    if angle > 65: return "Swing plane is too steep - shallow out your arc"
    return "Swing plane looks solid"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _safe_get(results, idx):
    if not results:
        return None
    return results[max(0, min(idx, len(results) - 1))]


def _joint_angle(a, b, c):
    ax, ay, az = a.x - b.x, a.y - b.y, a.z - b.z
    cx, cy, cz = c.x - b.x, c.y - b.y, c.z - b.z
    dot   = ax*cx + ay*cy + az*cz
    mag_a = math.sqrt(ax*ax + ay*ay + az*az)
    mag_c = math.sqrt(cx*cx + cy*cy + cz*cz)
    if mag_a < 1e-6 or mag_c < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_c)))))


def _tolerance_score(value, ideal, tolerance):
    deviation = abs(value - ideal)
    return round(max(0.0, 100.0 * (1.0 - deviation / (tolerance * 2))), 1)
