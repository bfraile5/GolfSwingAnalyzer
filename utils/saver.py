"""Save swing clips and score reports to disk."""
from __future__ import annotations
import json
import os
from datetime import datetime
import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def save_swing(
    frames_cam0: list[np.ndarray],
    frames_cam2: list[np.ndarray],
    metrics,
    phases,
    swing_analysis=None,
) -> str:
    """Save annotated MP4s and JSON report. Returns the save directory path."""
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.SAVE_DIR, timestamp)
    os.makedirs(save_path, exist_ok=True)

    _write_video(frames_cam0, os.path.join(save_path, "face_on.mp4"))
    _write_video(frames_cam2, os.path.join(save_path, "down_the_line.mp4"))
    _write_report(metrics, phases, swing_analysis, os.path.join(save_path, "report.json"))

    print(f"[Saver] Saved swing to {save_path}")
    return save_path


def _write_video(frames: list[np.ndarray], path: str) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, config.TARGET_FPS, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _write_report(metrics, phases, swing_analysis, path: str) -> None:
    data = {
        "scores": {
            "overall":        metrics.overall_score,
            "spine_tilt":     metrics.spine_angle_score,
            "hip_rotation":   metrics.hip_rotation_score,
            "knee_flex":      metrics.knee_flex_score,
            "head_stability": metrics.head_stability_score,
            "arm_extension":  metrics.arm_extension_score,
            "swing_plane":    metrics.swing_plane_score,
        },
        "raw_values": {
            "spine_angle_deg":  metrics.spine_angle_deg,
            "hip_rotation_deg": metrics.hip_rotation_deg,
            "knee_flex_deg":    metrics.knee_flex_deg,
            "head_stddev":      metrics.head_stddev,
            "arm_extension_deg": metrics.arm_extension_deg,
        },
        "tips": metrics.tips,
        "phases": {
            "p1_address_frame":           phases.p1,
            "p2_takeaway_frame":          phases.p2,
            "p3_backswing_frame":         phases.p3,
            "p4_top_frame":               phases.p4,
            "p5_downswing_frame":         phases.p5,
            "p6_pre_impact_frame":        phases.p6,
            "p7_impact_frame":            phases.p7,
            "p8_follow_through_frame":    phases.p8,
            "p9_extension_frame":         phases.p9,
            "p10_finish_frame":           phases.p10,
            "total_frames":               phases.total_frames,
        },
    }

    if swing_analysis is not None:
        data["swing_analysis"] = {
            "evaluations": {
                "address":        swing_analysis.address_evaluation,
                "backswing":      swing_analysis.backswing_evaluation,
                "transition":     swing_analysis.transition_evaluation,
                "impact":         swing_analysis.impact_evaluation,
                "follow_through": swing_analysis.follow_through_evaluation,
            },
            "summary": {
                "max_shoulder_rotation": round(swing_analysis.max_shoulder_rotation, 1),
                "max_hip_rotation":      round(swing_analysis.max_hip_rotation, 1),
            },
            "positions": [
                {
                    "label":                 p.label,
                    "frame_idx":             p.frame_idx,
                    "spine_angle":           round(p.spine_angle, 1),
                    "knee_flex":             round(p.knee_flex, 1),
                    "lead_arm_angle":        round(p.lead_arm_angle, 1),
                    "wrist_height_ratio":    round(p.wrist_height_ratio, 3),
                    "head_x":                round(p.head_x, 4),
                    "head_y":                round(p.head_y, 4),
                    "hip_rotation_delta":    round(p.hip_rotation_delta, 1),
                    "shoulder_rotation_delta": round(p.shoulder_rotation_delta, 1),
                    "head_drift":            round(p.head_drift, 4),
                }
                for p in swing_analysis.positions
            ],
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
