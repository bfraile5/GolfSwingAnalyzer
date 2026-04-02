"""Batch MediaPipe Pose processing for a captured clip."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

import mediapipe as mp

_mp_pose = mp.solutions.pose
_PoseLandmark = _mp_pose.PoseLandmark


@dataclass
class PoseResult:
    timestamp: float
    frame_index: int
    landmarks: object | None          # mp NormalizedLandmarkList
    world_landmarks: object | None    # mp LandmarkList (metric)
    detected: bool = False

    def lm(self, name: str):
        """Get a normalized landmark by name, or None."""
        if not self.detected or self.landmarks is None:
            return None
        idx = _PoseLandmark[name].value
        return self.landmarks.landmark[idx]

    def wlm(self, name: str):
        """Get a world (metric) landmark by name, or None."""
        if not self.detected or self.world_landmarks is None:
            return None
        idx = _PoseLandmark[name].value
        return self.world_landmarks.landmark[idx]


class PoseRunner:
    """Processes a list of BGR frames through MediaPipe Pose (Lite model).

    Create one instance per camera clip; do NOT share between threads.
    """

    def __init__(self) -> None:
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
        )

    def process_clip(
        self,
        frames: list[tuple[float, np.ndarray]],
        sample_fps: int = config.ANALYSIS_SAMPLE_FPS,
        capture_fps: int = config.TARGET_FPS,
    ) -> list[PoseResult]:
        """Process a subset of frames (downsampled to sample_fps).

        Returns one PoseResult per *input* frame (non-sampled frames get
        the nearest sampled result propagated forward for phase detection).
        """
        step = max(1, round(capture_fps / sample_fps))
        sampled_indices = list(range(0, len(frames), step))

        results_by_idx: dict[int, PoseResult] = {}
        for i in sampled_indices:
            ts, bgr = frames[i]
            rgb = bgr[:, :, ::-1].copy()  # BGR→RGB, contiguous
            mp_result = self._pose.process(rgb)
            detected = mp_result.pose_landmarks is not None
            results_by_idx[i] = PoseResult(
                timestamp=ts,
                frame_index=i,
                landmarks=mp_result.pose_landmarks,
                world_landmarks=mp_result.pose_world_landmarks,
                detected=detected,
            )

        # Build full-length list, propagating nearest sampled result
        full_results: list[PoseResult] = []
        last_result = results_by_idx.get(0)
        for i in range(len(frames)):
            if i in results_by_idx:
                last_result = results_by_idx[i]
            ts, _ = frames[i]
            full_results.append(
                PoseResult(
                    timestamp=ts,
                    frame_index=i,
                    landmarks=last_result.landmarks if last_result else None,
                    world_landmarks=last_result.world_landmarks if last_result else None,
                    detected=last_result.detected if last_result else False,
                )
            )
        return full_results

    def close(self) -> None:
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
