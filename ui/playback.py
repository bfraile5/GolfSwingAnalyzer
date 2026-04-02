"""Video player for looping annotated swing clip playback, with timeline and P-detail."""
from __future__ import annotations
import pygame
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class VideoPlayer:
    """Plays two annotated clips side-by-side.

    New in this version
    -------------------
    * P1–P10 scrub timeline below the video panels
    * Per-position body-angle detail strip (spine, knee, hip Δ, shoulder Δ, etc.)
    * Play / pause state; step-forward / step-back
    * Click anywhere on the scrub bar to seek
    """

    def __init__(
        self,
        frames_cam0: list[np.ndarray],
        frames_cam2: list[np.ndarray],
        phases,
        swing_analysis=None,
        loop: bool = True,
        fonts: dict | None = None,
    ) -> None:
        self._phases        = phases
        self._swing_analysis = swing_analysis
        self._loop          = loop
        self._playing       = True
        self._speed         = 1.0
        self._frame_idx     = 0
        self._accum_ms      = 0.0
        self._frame_duration_ms = 1000.0 / config.TARGET_FPS
        self._fonts         = fonts or _default_fonts()

        # Scrub-bar rect — set on first draw so we can hit-test clicks
        self._timeline_bar_rect: pygame.Rect | None = None

        dw = config.VIDEO_DISPLAY_WIDTH
        dh = config.VIDEO_DISPLAY_HEIGHT
        self._surfaces_cam0 = _preconvert(frames_cam0, dw, dh)
        self._surfaces_cam2 = _preconvert(frames_cam2, dw, dh)
        self._total_frames  = max(
            len(self._surfaces_cam0), len(self._surfaces_cam2), 1
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, dt_ms: float) -> None:
        """Advance the frame pointer by elapsed time (no-op when paused)."""
        if not self._playing:
            return
        self._accum_ms += dt_ms * self._speed
        n = int(self._accum_ms / self._frame_duration_ms)
        if n > 0:
            self._accum_ms -= n * self._frame_duration_ms
            self._frame_idx += n
            if self._frame_idx >= self._total_frames:
                if self._loop:
                    self._frame_idx = 0
                else:
                    self._frame_idx = self._total_frames - 1

    def handle_events(self, events: list) -> None:
        """Handle pre-drained pygame events (timeline seeks)."""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if (self._timeline_bar_rect
                        and self._timeline_bar_rect.collidepoint(event.pos)):
                    rel  = event.pos[0] - self._timeline_bar_rect.x
                    frac = rel / max(1, self._timeline_bar_rect.width)
                    self._frame_idx = int(frac * self._total_frames)
                    self._frame_idx = max(0, min(self._total_frames - 1, self._frame_idx))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw video panels, timeline scrub bar, and P-detail strip."""
        self._draw_video(surface)
        self._draw_timeline(surface)
        self._draw_p_detail(surface)

    def reset(self) -> None:
        self._frame_idx = 0
        self._accum_ms  = 0.0
        self._playing   = True

    def set_speed(self, factor: float) -> None:
        self._speed = max(0.1, min(4.0, factor))

    def toggle_play(self) -> None:
        self._playing = not self._playing

    def step_forward(self, frames: int = 1) -> None:
        self._playing    = False
        self._frame_idx  = min(self._total_frames - 1, self._frame_idx + frames)

    def step_back(self, frames: int = 1) -> None:
        self._playing    = False
        self._frame_idx  = max(0, self._frame_idx - frames)

    def jump_to_p(self, n: int) -> None:
        """Jump to P-position n (1-based, 1–10)."""
        if self._phases and 1 <= n <= 10:
            self._frame_idx = self._phases.p_frame(n)

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def current_phase(self) -> str:
        if self._phases:
            return self._phases.phase_for_frame(self._frame_idx)
        return ""

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw_video(self, surface: pygame.Surface) -> None:
        y_start = config.HEADER_HEIGHT
        cell_w  = config.VIDEO_CELL_WIDTH
        cell_h  = config.REVIEW_VIDEO_HEIGHT

        cam_labels = ["Face-On  (CAM 0)", "Down-the-Line  (CAM 2)"]

        for i, surfs in enumerate([self._surfaces_cam0, self._surfaces_cam2]):
            cell_x = i * cell_w
            pygame.draw.rect(surface, config.COLOR_PANEL,
                             (cell_x, y_start, cell_w, cell_h))
            if surfs:
                idx = min(self._frame_idx, len(surfs) - 1)
                s   = surfs[idx]
                fx  = cell_x + (cell_w - s.get_width())  // 2
                fy  = y_start + (cell_h - s.get_height()) // 2
                surface.blit(s, (fx, fy))
            else:
                msg = self._fonts["medium"].render(
                    f"No {cam_labels[i]} Camera", True, config.COLOR_TEXT_DIM
                )
                surface.blit(msg, (
                    cell_x + (cell_w - msg.get_width()) // 2,
                    y_start + cell_h // 2,
                ))

            # Camera label (top-left of each cell)
            lbl = self._fonts["small"].render(cam_labels[i], True, config.COLOR_TEXT_DIM)
            surface.blit(lbl, (cell_x + 10, y_start + 8))

        # Centre divider
        pygame.draw.line(
            surface, config.COLOR_BORDER,
            (cell_w, y_start), (cell_w, y_start + cell_h), 1
        )

        # P-position badge — top-right of video area, semi-transparent background
        if self._phases:
            p_num = self._nearest_p_number()
            if p_num:
                badge_text = self._phases.p_label(p_num)
                badge_surf = self._fonts["large"].render(badge_text, True, config.COLOR_ACCENT)
                bx = config.DISPLAY_WIDTH - badge_surf.get_width() - 18
                by = y_start + 10
                bg = pygame.Surface(
                    (badge_surf.get_width() + 18, badge_surf.get_height() + 10),
                    pygame.SRCALPHA,
                )
                bg.fill((0, 0, 0, 160))
                surface.blit(bg, (bx - 9, by - 5))
                surface.blit(badge_surf, (bx, by))

    def _draw_timeline(self, surface: pygame.Surface) -> None:
        """Scrub bar: progress track + P1–P10 tick marks + playhead."""
        tl_y = config.HEADER_HEIGHT + config.REVIEW_VIDEO_HEIGHT
        tl_w = config.DISPLAY_WIDTH
        tl_h = config.TIMELINE_HEIGHT

        # Panel background + top border
        pygame.draw.rect(surface, config.COLOR_PANEL, (0, tl_y, tl_w, tl_h))
        pygame.draw.line(surface, config.COLOR_BORDER,
                         (0, tl_y), (tl_w, tl_y), 1)

        # Track geometry
        margin_l = 58
        margin_r = 90
        track_x  = margin_l
        track_w  = tl_w - margin_l - margin_r
        track_y  = tl_y + tl_h - 20   # pushed toward bottom so labels sit above
        track_h  = 8

        # Hit area for clicks spans the full panel height (easier to tap)
        self._timeline_bar_rect = pygame.Rect(track_x, tl_y + 4, track_w, tl_h - 8)

        # Track background
        pygame.draw.rect(surface, config.COLOR_BG,
                         (track_x, track_y, track_w, track_h), border_radius=4)

        # Filled portion (progress)
        if self._total_frames > 1:
            fill_w = int(track_w * self._frame_idx / (self._total_frames - 1))
            if fill_w > 0:
                pygame.draw.rect(surface, config.COLOR_ACCENT,
                                 (track_x, track_y, fill_w, track_h), border_radius=4)

        # P1–P10 tick marks and labels
        if self._phases and self._total_frames > 1:
            nearest = self._nearest_p_number()
            for n in range(1, 11):
                fidx  = self._phases.p_frame(n)
                frac  = fidx / (self._total_frames - 1)
                tx    = int(track_x + frac * track_w)
                is_cur = (nearest == n)
                color = config.COLOR_ACCENT if is_cur else config.COLOR_TEXT_DIM
                tick_top = track_y - (16 if is_cur else 11)
                pygame.draw.line(surface, color,
                                 (tx, tick_top), (tx, track_y + track_h + 3), 2)
                lbl = self._fonts["small"].render(f"P{n}", True, color)
                lx  = tx - lbl.get_width() // 2
                lx  = max(track_x, min(lx, track_x + track_w - lbl.get_width()))
                surface.blit(lbl, (lx, tl_y + 5))

        # Playhead circle
        if self._total_frames > 1:
            ph_x = int(track_x + track_w * self._frame_idx / (self._total_frames - 1))
            pygame.draw.circle(surface, config.COLOR_BORDER,
                               (ph_x, track_y + track_h // 2), 9)
            pygame.draw.circle(surface, config.COLOR_TEXT,
                               (ph_x, track_y + track_h // 2), 7)

        # Play/pause icon (left margin)
        ix = 16
        iy = tl_y + tl_h // 2
        if self._playing:
            # Pause: two vertical bars
            pygame.draw.rect(surface, config.COLOR_TEXT, (ix,      iy - 9, 5, 18))
            pygame.draw.rect(surface, config.COLOR_TEXT, (ix + 10, iy - 9, 5, 18))
        else:
            # Play: filled triangle
            pts = [(ix, iy - 10), (ix, iy + 10), (ix + 18, iy)]
            pygame.draw.polygon(surface, config.COLOR_TEXT, pts)

        # Frame counter (right margin)
        fc = self._fonts["small"].render(
            f"{self._frame_idx + 1} / {self._total_frames}",
            True, config.COLOR_TEXT_DIM,
        )
        surface.blit(fc, (
            tl_w - fc.get_width() - 10,
            tl_y + (tl_h - fc.get_height()) // 2,
        ))

    def _draw_p_detail(self, surface: pygame.Surface) -> None:
        """Body-angle detail strip for the nearest P-position."""
        pd_y = config.HEADER_HEIGHT + config.REVIEW_VIDEO_HEIGHT + config.TIMELINE_HEIGHT
        pd_w = config.DISPLAY_WIDTH
        pd_h = config.P_DETAIL_HEIGHT

        pygame.draw.rect(surface, (18, 18, 40), (0, pd_y, pd_w, pd_h))
        pygame.draw.line(surface, config.COLOR_BORDER,
                         (0, pd_y), (pd_w, pd_y), 1)

        p_num = self._nearest_p_number()
        if not (self._swing_analysis and self._phases and p_num):
            # No data — show a subtle hint
            hint = self._fonts["small"].render(
                "No position data available", True, config.COLOR_TEXT_DIM
            )
            surface.blit(hint, (20, pd_y + (pd_h - hint.get_height()) // 2))
            return

        pos        = self._swing_analysis.positions[p_num - 1]
        phase_text = self._phases.p_label(p_num)   # e.g. "P4: Top"

        # Left: position label
        lbl = self._fonts["medium"].render(phase_text.upper(), True, config.COLOR_ACCENT)
        lbl_x = 18
        surface.blit(lbl, (lbl_x, pd_y + (pd_h - lbl.get_height()) // 2))

        # Six metric columns
        # Each tuple: (display_name, formatted_value, raw_value, (ideal_lo, ideal_hi))
        metrics = [
            ("Spine Tilt",   f"{pos.spine_angle:.0f}°",
             pos.spine_angle, (25.0, 50.0)),
            ("Knee Flex",    f"{pos.knee_flex:.0f}°",
             pos.knee_flex, (148.0, 170.0)),
            ("Hip Rot Δ",    f"{pos.hip_rotation_delta:.0f}°",
             pos.hip_rotation_delta, (0.0, 90.0)),
            ("Shoulder Δ",   f"{pos.shoulder_rotation_delta:.0f}°",
             pos.shoulder_rotation_delta, (0.0, 120.0)),
            ("Lead Arm",     f"{pos.lead_arm_angle:.0f}°",
             pos.lead_arm_angle, (148.0, 180.0)),
            ("Head Drift",   f"{pos.head_drift:.3f}",
             pos.head_drift, (0.0, 0.07)),
        ]

        label_block_w = lbl.get_width() + 34
        avail_w       = pd_w - label_block_w - 8
        col_w         = avail_w // len(metrics)

        for i, (name, val_str, raw_val, (lo, hi)) in enumerate(metrics):
            mx = label_block_w + i * col_w
            my = pd_y + 10

            # Divider between columns
            if i > 0:
                pygame.draw.line(
                    surface, config.COLOR_BORDER,
                    (mx - 4, pd_y + 10), (mx - 4, pd_y + pd_h - 10), 1
                )

            # Metric name (dim)
            name_s = self._fonts["small"].render(name, True, config.COLOR_TEXT_DIM)
            surface.blit(name_s, (mx, my))

            # Value — green if in ideal range, amber otherwise
            in_range  = lo <= raw_val <= hi
            val_color = config.COLOR_GOOD if in_range else config.COLOR_WARN
            val_s = self._fonts["medium"].render(val_str, True, val_color)
            surface.blit(val_s, (mx, my + name_s.get_height() + 4))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _nearest_p_number(self) -> int | None:
        """P-position number (1–10) closest to the current frame."""
        if not self._phases or self._total_frames <= 1:
            return None
        best_n, best_d = 1, abs(self._frame_idx - self._phases.p_frame(1))
        for n in range(2, 11):
            d = abs(self._frame_idx - self._phases.p_frame(n))
            if d < best_d:
                best_d, best_n = d, n
        return best_n


# ── Module helpers ─────────────────────────────────────────────────────────────

def _preconvert(
    frames: list[np.ndarray], dw: int, dh: int
) -> list[pygame.Surface]:
    surfs = []
    for frame in frames:
        rgb     = frame[:, :, ::-1].copy()
        resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        surfs.append(pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2))))
    return surfs


def _default_fonts() -> dict:
    pygame.font.init()
    try:
        def f(size): return pygame.font.SysFont("liberationsans", size)
    except Exception:
        def f(size): return pygame.font.Font(None, size)
    return {
        "title":  f(28),
        "large":  f(32),
        "medium": f(20),
        "small":  f(16),
        "score":  f(32),
        "metric": f(20),
    }
