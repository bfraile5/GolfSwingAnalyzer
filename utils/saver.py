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
) -> str:
    """Save annotated MP4s and JSON report. Returns the save directory path."""
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.SAVE_DIR, timestamp)
    os.makedirs(save_path, exist_ok=True)

    _write_video(frames_cam0, os.path.join(save_path, "face_on.mp4"))
    _write_video(frames_cam2, os.path.join(save_path, "down_the_line.mp4"))
    _write_report(metrics, phases, os.path.join(save_path, "report.json"))

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


def _write_report(metrics, phases, path: str) -> None:
    data = {
        "scores": {
            "overall": metrics.overall_score,
            "spine_tilt": metrics.spine_angle_score,
            "hip_rotation": metrics.hip_rotation_score,
            "knee_flex": metrics.knee_flex_score,
            "head_stability": metrics.head_stability_score,
            "arm_extension": metrics.arm_extension_score,
            "swing_plane": metrics.swing_plane_score,
        },
        "raw_values": {
            "spine_angle_deg": metrics.spine_angle_deg,
            "hip_rotation_deg": metrics.hip_rotation_deg,
            "knee_flex_deg": metrics.knee_flex_deg,
            "head_stddev": metrics.head_stddev,
            "arm_extension_deg": metrics.arm_extension_deg,
        },
        "tips": metrics.tips,
        "phases": {
            "address_frame": phases.address_idx,
            "backswing_top_frame": phases.backswing_top_idx,
            "impact_frame": phases.impact_idx,
            "follow_through_frame": phases.follow_through_idx,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
