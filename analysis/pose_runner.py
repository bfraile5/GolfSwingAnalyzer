"""Batch MediaPipe Pose processing with temporal smoothing for a captured clip."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

import mediapipe as mp

_mp_pose = mp.solutions.pose
_PoseLandmark = _mp_pose.PoseLandmark

_N_LANDMARKS = 33

# Gaussian smoothing parameters (applied along the time axis)
_SMOOTH_SIGMA = 2.0    # std-dev in frames
_SMOOTH_HALF = 4       # half-width → kernel size = 2*4+1 = 9 frames
_VIS_THRESHOLD = 0.35  # below this, treat the landmark as missing for interpolation


@dataclass
class SmoothLandmark:
    """Lightweight landmark with the same attribute interface as a MediaPipe landmark."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class PoseResult:
    timestamp: float
    frame_index: int
    landmarks: object | None          # mp NormalizedLandmarkList (raw, for skeleton drawing)
    world_landmarks: object | None    # mp LandmarkList (metric, raw)
    detected: bool = False
    # Smoothed per-frame arrays shape = (33, 4)  →  [x, y, z, visibility]
    smoothed_norm: np.ndarray | None = field(default=None, repr=False)
    smoothed_world: np.ndarray | None = field(default=None, repr=False)

    def lm(self, name: str) -> SmoothLandmark | None:
        """Normalised landmark by name — returns smoothed value when available."""
        idx = _PoseLandmark[name].value
        if self.smoothed_norm is not None:
            x, y, z, v = self.smoothed_norm[idx]
            if v > 0.01:   # landmark was detected in at least one nearby frame
                return SmoothLandmark(float(x), float(y), float(z), float(v))
        if not self.detected or self.landmarks is None:
            return None
        return self.landmarks.landmark[idx]  # type: ignore[return-value]

    def wlm(self, name: str) -> SmoothLandmark | None:
        """World (metric) landmark by name — returns smoothed value when available."""
        idx = _PoseLandmark[name].value
        if self.smoothed_world is not None:
            x, y, z, v = self.smoothed_world[idx]
            if v > 0.01:
                return SmoothLandmark(float(x), float(y), float(z), float(v))
        if not self.detected or self.world_landmarks is None:
            return None
        return self.world_landmarks.landmark[idx]  # type: ignore[return-value]


class PoseRunner:
    """Runs MediaPipe Pose on every frame in a clip and applies temporal smoothing.

    Strategy
    --------
    1. Process every frame through MediaPipe (no downsampling).
    2. Extract per-frame landmark arrays (n_frames, 33, 4).
    3. For each landmark: linearly interpolate gaps caused by low confidence, then
       convolve with a Gaussian kernel along the time axis.
    4. Attach the smoothed arrays to each PoseResult for downstream analysis.
       The raw MediaPipe landmark lists are preserved for skeleton overlay drawing.

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
        """Process ALL frames through MediaPipe Pose with Gaussian temporal smoothing.

        The ``sample_fps`` / ``capture_fps`` parameters are retained for API
        compatibility but are no longer used for downsampling — every frame is
        processed.  Raw MediaPipe landmarks are kept on each ``PoseResult`` for
        skeleton drawing; smoothed arrays are attached for downstream analysis.
        """
        if not frames:
            return []

        n = len(frames)

        # ── Pass 1: run MediaPipe on every frame ───────────────────────────────
        raw: list[PoseResult] = []
        for i, (ts, bgr) in enumerate(frames):
            rgb = bgr[:, :, ::-1].copy()   # BGR → RGB, contiguous
            mp_res = self._pose.process(rgb)
            detected = mp_res.pose_landmarks is not None
            raw.append(PoseResult(
                timestamp=ts,
                frame_index=i,
                landmarks=mp_res.pose_landmarks,
                world_landmarks=mp_res.pose_world_landmarks,
                detected=detected,
            ))

        # ── Pass 2: extract raw landmark arrays ───────────────────────────────
        raw_norm = np.zeros((n, _N_LANDMARKS, 4), dtype=np.float32)
        raw_world = np.zeros((n, _N_LANDMARKS, 4), dtype=np.float32)
        det = np.array([r.detected for r in raw], dtype=bool)

        for i, pr in enumerate(raw):
            if pr.detected and pr.landmarks is not None:
                for j, lm in enumerate(pr.landmarks.landmark):
                    raw_norm[i, j] = (lm.x, lm.y, lm.z, lm.visibility)
                for j, wlm in enumerate(pr.world_landmarks.landmark):
                    raw_world[i, j] = (wlm.x, wlm.y, wlm.z, wlm.visibility)

        # ── Pass 3: temporal smoothing ─────────────────────────────────────────
        sm_norm, sm_world = _smooth_landmark_arrays(raw_norm, raw_world, det)

        # ── Pass 4: attach smoothed arrays to PoseResult objects ───────────────
        final: list[PoseResult] = []
        for i, pr in enumerate(raw):
            # Mark detected=True if smoothing borrowed data from neighbour frames
            # (check interpolated visibility on the nose landmark, index 0).
            has_smooth = bool(sm_norm[i, 0, 3] > 0.01)
            final.append(PoseResult(
                timestamp=pr.timestamp,
                frame_index=i,
                landmarks=pr.landmarks,
                world_landmarks=pr.world_landmarks,
                detected=pr.detected or has_smooth,
                smoothed_norm=sm_norm[i],
                smoothed_world=sm_world[i],
            ))

        return final

    def close(self) -> None:
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Temporal smoothing helpers ─────────────────────────────────────────────────

def _gaussian_kernel(sigma: float, half_width: int) -> np.ndarray:
    """Build a normalised 1-D Gaussian kernel."""
    x = np.arange(-half_width, half_width + 1, dtype=np.float32)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _smooth_landmark_arrays(
    raw_norm: np.ndarray,   # (n, 33, 4)
    raw_world: np.ndarray,  # (n, 33, 4)
    detected: np.ndarray,   # (n,) bool
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian temporal smoothing over the frame axis for every landmark.

    Per-landmark algorithm:
    1. Identify frames where detection confidence meets the threshold.
    2. Linearly interpolate x/y/z/visibility for low-confidence frames from
       their nearest well-detected neighbours.
    3. Convolve x/y/z with a Gaussian kernel; visibility is only interpolated
       (not further smoothed) so callers can still inspect raw confidence.
    """
    n = raw_norm.shape[0]
    kernel = _gaussian_kernel(_SMOOTH_SIGMA, _SMOOTH_HALF)
    t = np.arange(n, dtype=np.float32)

    sm_norm = raw_norm.copy()
    sm_world = raw_world.copy()

    for lm_idx in range(_N_LANDMARKS):
        vis = raw_norm[:, lm_idx, 3]
        good = (vis >= _VIS_THRESHOLD) & detected
        if good.sum() < 2:
            continue
        good_t = t[good]

        for src, dst in [(raw_norm, sm_norm), (raw_world, sm_world)]:
            # Smooth x, y, z
            for coord in range(3):
                series = src[:, lm_idx, coord].copy()
                if not np.all(good):
                    series = np.interp(t, good_t, series[good])
                dst[:, lm_idx, coord] = np.convolve(series, kernel, mode='same')
            # Interpolate visibility (fills gaps so lm() can detect valid data)
            vis_series = src[:, lm_idx, 3].copy()
            if not np.all(good):
                vis_series = np.interp(t, good_t, vis_series[good])
            dst[:, lm_idx, 3] = vis_series

    return sm_norm, sm_world
