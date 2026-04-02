"""Video player for looping annotated swing clip playback."""
from __future__ import annotations
import time
import numpy as np
import pygame
import cv2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class VideoPlayer:
    """Plays two annotated clips side-by-side in the video area."""

    def __init__(
        self,
        frames_cam0: list[np.ndarray],
        frames_cam2: list[np.ndarray],
        phases,
        loop: bool = True,
    ) -> None:
        self._phases = phases
        self._loop = loop
        self._speed = 1.0
        self._frame_idx = 0
        self._accum_ms = 0.0
        self._frame_duration_ms = 1000.0 / config.TARGET_FPS

        # Pre-convert all frames to pygame surfaces once
        self._surfaces_cam0 = self._preconvert(frames_cam0)
        self._surfaces_cam2 = self._preconvert(frames_cam2)
        self._total_frames = max(len(self._surfaces_cam0), len(self._surfaces_cam2))

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, dt_ms: float) -> None:
        """Advance frame pointer by elapsed time."""
        self._accum_ms += dt_ms * self._speed
        frames_to_advance = int(self._accum_ms / self._frame_duration_ms)
        if frames_to_advance > 0:
            self._accum_ms -= frames_to_advance * self._frame_duration_ms
            self._frame_idx += frames_to_advance
            if self._frame_idx >= self._total_frames:
                if self._loop:
                    self._frame_idx = 0
                else:
                    self._frame_idx = self._total_frames - 1

    def draw(self, surface: pygame.Surface) -> None:
        """Blit current frame pair into the video area."""
        y_start = config.HEADER_HEIGHT
        cell_w = config.VIDEO_CELL_WIDTH
        cell_h = config.VIDEO_AREA_HEIGHT

        for i, surfaces in enumerate([self._surfaces_cam0, self._surfaces_cam2]):
            cell_x = i * cell_w
            if not surfaces:
                continue
            idx = min(self._frame_idx, len(surfaces) - 1)
            surf = surfaces[idx]
            fx = cell_x + (cell_w - surf.get_width()) // 2
            fy = y_start + (cell_h - surf.get_height()) // 2
            surface.blit(surf, (fx, fy))

        # Divider
        pygame.draw.line(
            surface, config.COLOR_BORDER,
            (cell_w, y_start),
            (cell_w, y_start + cell_h), 1
        )

        # Camera labels
        # (drawn by screen.py header — nothing extra needed here)

    def reset(self) -> None:
        self._frame_idx = 0
        self._accum_ms = 0.0

    def set_speed(self, factor: float) -> None:
        self._speed = max(0.1, min(4.0, factor))

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def current_phase(self) -> str:
        if self._phases:
            return self._phases.phase_for_frame(self._frame_idx)
        return ""

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _preconvert(frames: list[np.ndarray]) -> list[pygame.Surface]:
        """Convert BGR numpy frames to pygame Surfaces once at load time."""
        surfaces = []
        dw, dh = config.VIDEO_DISPLAY_WIDTH, config.VIDEO_DISPLAY_HEIGHT
        for frame in frames:
            rgb = frame[:, :, ::-1].copy()  # BGR→RGB
            resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
            surf = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
            surfaces.append(surf)
        return surfaces
