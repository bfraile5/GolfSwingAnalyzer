"""Draw pose skeleton overlay on BGR frames."""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from analysis.pose_runner import PoseResult

_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing spec using our cyan colour
_LANDMARK_SPEC = _mp_drawing.DrawingSpec(
    color=config.COLOR_POSE_SKELETON, thickness=2, circle_radius=3
)
_CONNECTION_SPEC = _mp_drawing.DrawingSpec(
    color=config.COLOR_POSE_SKELETON, thickness=2
)


def draw_pose_overlay(frame: np.ndarray, result: PoseResult) -> np.ndarray:
    """Draw skeleton on a copy of frame. Returns annotated BGR frame."""
    out = frame.copy()
    if result.detected and result.landmarks is not None:
        _mp_drawing.draw_landmarks(
            out,
            result.landmarks,
            _mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=_LANDMARK_SPEC,
            connection_drawing_spec=_CONNECTION_SPEC,
        )
    return out


def draw_phase_label(frame: np.ndarray, label: str, color: tuple = (200, 200, 200)) -> np.ndarray:
    """Draw phase name at bottom of frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = (w - tw) // 2
    y = h - 15
    # Shadow
    cv2.putText(out, label, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(out, label, (x, y), font, scale, color, thickness)
    return out


def annotate_clip(
    frames: list[tuple[float, np.ndarray]],
    results: list[PoseResult],
    phases,
) -> list[np.ndarray]:
    """Return list of fully annotated BGR frames (pose + phase label)."""
    annotated = []
    for i, (ts, frame) in enumerate(frames):
        result = results[i] if i < len(results) else None
        if result:
            out = draw_pose_overlay(frame, result)
        else:
            out = frame.copy()
        if phases:
            label = phases.phase_for_frame(i)
            out = draw_phase_label(out, label)
        annotated.append(out)
    return annotated
