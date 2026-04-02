"""Compute swing metrics from pose results. Pure functions, no I/O."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from analysis.pose_runner import PoseResult
from analysis.phases import SwingPhases

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


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

    m.spine_angle_deg, m.spine_angle_score = _spine_angle(results_cam0, phases.address_idx)
    m.tips["Spine Tilt"] = _spine_tip(m.spine_angle_deg)

    m.hip_rotation_deg, m.hip_rotation_score = _hip_rotation(
        results_cam0, phases.address_idx, phases.impact_idx)
    m.tips["Hip Rotation"] = _hip_tip(m.hip_rotation_deg)

    m.knee_flex_deg, m.knee_flex_score = _knee_flex(results_cam0, phases.address_idx)
    m.tips["Knee Flex"] = _knee_tip(m.knee_flex_deg)

    m.head_stddev, m.head_stability_score = _head_stability(
        results_cam0, phases.address_idx, phases.impact_idx)
    m.tips["Head Stability"] = _head_tip(m.head_stddev)

    m.arm_extension_deg, m.arm_extension_score = _arm_extension(
        results_cam2, phases.impact_idx)
    m.tips["Arm Extension"] = _arm_tip(m.arm_extension_deg)

    plane_angle, m.swing_plane_score = _swing_plane(
        results_cam2, phases.address_idx, phases.backswing_top_idx)
    m.tips["Swing Plane"] = _plane_tip(plane_angle)

    scores = [
        m.spine_angle_score, m.hip_rotation_score, m.knee_flex_score,
        m.head_stability_score, m.arm_extension_score, m.swing_plane_score,
    ]
    m.overall_score = round(sum(scores) / len(scores), 1)
    return m


# ── Individual metric functions ───────────────────────────────────────────────

def _spine_angle(results, address_idx):
    r = _safe_get(results, address_idx)
    ls = r.wlm("LEFT_SHOULDER") if r else None
    lh = r.wlm("LEFT_HIP") if r else None
    rs = r.wlm("RIGHT_SHOULDER") if r else None
    rh = r.wlm("RIGHT_HIP") if r else None
    if None in (ls, lh, rs, rh):
        return 0.0, 50.0
    sx = (ls.x + rs.x) / 2
    sy = (ls.y + rs.y) / 2
    sz = (ls.z + rs.z) / 2
    hx = (lh.x + rh.x) / 2
    hy = (lh.y + rh.y) / 2
    hz = (lh.z + rh.z) / 2
    dx = sx - hx; dy = sy - hy; dz = sz - hz
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-6:
        return 0.0, 50.0
    cos_theta = dy / length
    angle_from_vertical = math.degrees(math.acos(max(-1, min(1, cos_theta))))
    score = _tolerance_score(angle_from_vertical, config.SPINE_ANGLE_IDEAL, config.SPINE_ANGLE_TOLERANCE)
    return round(angle_from_vertical, 1), score


def _hip_rotation(results, address_idx, impact_idx):
    def hip_angle(r):
        if r is None:
            return None
        lh = r.wlm("LEFT_HIP")
        rh = r.wlm("RIGHT_HIP")
        if lh is None or rh is None:
            return None
        return math.degrees(math.atan2(rh.z - lh.z, rh.x - lh.x))

    r_addr = _safe_get(results, address_idx)
    r_imp = _safe_get(results, impact_idx)
    a1 = hip_angle(r_addr)
    a2 = hip_angle(r_imp)
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
    angle = _joint_angle(hip, knee, ankle)
    ideal_mid = (config.KNEE_FLEX_IDEAL_MIN + config.KNEE_FLEX_IDEAL_MAX) / 2
    tolerance = (config.KNEE_FLEX_IDEAL_MAX - config.KNEE_FLEX_IDEAL_MIN) / 2
    score = _tolerance_score(angle, ideal_mid, tolerance)
    return round(angle, 1), score


def _head_stability(results, address_idx, impact_idx):
    xs, ys = [], []
    for r in results[address_idx:impact_idx + 1]:
        nose = r.lm("NOSE")
        if nose:
            xs.append(nose.x)
            ys.append(nose.y)
    if len(xs) < 2:
        return 0.0, 50.0
    stddev = float(np.std(xs) + np.std(ys))
    score = max(0.0, 100 * (1 - stddev / config.HEAD_STABILITY_MAX_STDDEV))
    return round(stddev, 4), round(score, 1)


def _arm_extension(results, impact_idx):
    r = _safe_get(results, impact_idx)
    if r is None:
        return 0.0, 50.0
    shoulder = r.wlm("LEFT_SHOULDER"); elbow = r.wlm("LEFT_ELBOW"); wrist = r.wlm("LEFT_WRIST")
    if None in (shoulder, elbow, wrist):
        shoulder = r.wlm("RIGHT_SHOULDER"); elbow = r.wlm("RIGHT_ELBOW"); wrist = r.wlm("RIGHT_WRIST")
    if None in (shoulder, elbow, wrist):
        return 0.0, 50.0
    angle = _joint_angle(shoulder, elbow, wrist)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_get(results, idx):
    if not results:
        return None
    return results[max(0, min(idx, len(results) - 1))]


def _joint_angle(a, b, c):
    ax, ay, az = a.x - b.x, a.y - b.y, a.z - b.z
    cx, cy, cz = c.x - b.x, c.y - b.y, c.z - b.z
    dot = ax*cx + ay*cy + az*cz
    mag_a = math.sqrt(ax*ax + ay*ay + az*az)
    mag_c = math.sqrt(cx*cx + cy*cy + cz*cz)
    if mag_a < 1e-6 or mag_c < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_c)))))


def _tolerance_score(value, ideal, tolerance):
    deviation = abs(value - ideal)
    return round(max(0.0, 100.0 * (1.0 - deviation / (tolerance * 2))), 1)
