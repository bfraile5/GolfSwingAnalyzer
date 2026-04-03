"""Main pygame display manager and 60 Hz render loop."""
from __future__ import annotations
import math
import time
import pygame
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from ui.playback import VideoPlayer
from ui.metrics_panel import MetricsPanel
from ui.controls import ControlBar


class Screen:
    """Owns the pygame window and dispatches rendering based on app state."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Golf Swing Analyzer")

        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        try:
            self.surface = pygame.display.set_mode(
                (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), flags
            )
        except Exception:
            self.surface = pygame.display.set_mode(
                (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            )

        self._clock  = pygame.time.Clock()
        self._fonts  = _load_fonts()

        # Review-mode objects
        self._player:         VideoPlayer  | None = None
        self._metrics_panel:  MetricsPanel | None = None
        self._controls = ControlBar(self.surface, self._fonts)

        # Buffering view state
        self._record_pulse = 0.0

        # Pre-allocated semi-transparent overlay for countdown (avoids per-frame
        # 8 MB SRCALPHA allocation which is very slow / broken on Pi display drivers)
        self._countdown_overlay = pygame.Surface(
            (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        )
        self._countdown_overlay.set_alpha(160)
        self._countdown_overlay.fill((0, 0, 0))

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_review(self, report) -> None:
        """Prepare video player and metrics panel from a SwingReport."""
        self._player = VideoPlayer(
            report.frames_cam0,
            report.frames_cam2,
            report.phases,
            swing_analysis=report.swing_analysis,
            fonts=self._fonts,
        )
        self._metrics_panel = MetricsPanel(report.metrics, self._fonts)

    def render_splash(self) -> list[str]:
        self.surface.fill(config.COLOR_BG)
        self._draw_header("Golf Swing Analyzer", "READY", None)
        msg  = self._fonts["large"].render("Preparing cameras…", True, config.COLOR_TEXT_DIM)
        x    = (config.DISPLAY_WIDTH  - msg.get_width())  // 2
        y    = config.DISPLAY_HEIGHT // 2 - msg.get_height() // 2
        self.surface.blit(msg, (x, y))
        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)
        return self._drain_events()

    def render_buffering(
        self,
        frame0: np.ndarray | None,
        frame2: np.ndarray | None,
        audio_available: bool,
    ) -> list[str]:
        self.surface.fill(config.COLOR_BG)
        self._draw_header("Golf Swing Analyzer", "RECORDING", config.COLOR_RECORD_DOT)
        self._draw_dual_preview(frame0, frame2)
        self._draw_buffering_instructions(audio_available)
        self._controls.draw_buffering()
        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)
        self._record_pulse += 0.08
        return self._drain_events()

    def render_countdown(
        self,
        seconds_remaining: float,
        frame0: np.ndarray | None,
        frame2: np.ndarray | None,
    ) -> list[str]:
        """Render countdown overlay on top of the live dual-camera preview."""
        self.surface.fill(config.COLOR_BG)
        self._draw_header("Golf Swing Analyzer", "GET READY", config.COLOR_WARN)
        self._draw_dual_preview(frame0, frame2)
        self._draw_countdown_overlay(seconds_remaining)
        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)
        return self._drain_events()

    def render_manual_recording(
        self,
        elapsed: float,
        total: float,
    ) -> list[str]:
        """Render the 'Recording…' screen during manual 10-second capture."""
        self.surface.fill(config.COLOR_BG)
        self._draw_header("Golf Swing Analyzer", "RECORDING", config.COLOR_RECORD_DOT)
        self._draw_manual_recording_overlay(elapsed, total)
        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)
        self._record_pulse += 0.08
        return self._drain_events()

    def render_analyzing(self, progress: float) -> list[str]:
        self.surface.fill(config.COLOR_BG)
        self._draw_header("Golf Swing Analyzer", "ANALYZING", config.COLOR_WARN)
        self._draw_analyzing_overlay(progress)
        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)
        return self._drain_events()

    def render_review(self, dt_ms: float) -> list[str]:
        """Render review screen — dual video + timeline + P-detail + metrics."""
        self.surface.fill(config.COLOR_BG)

        if self._player:
            self._player.update(dt_ms)
            phase = self._player.current_phase
            self._draw_header(
                "Golf Swing Analyzer",
                f"REVIEW  —  {phase.upper()}",
                config.COLOR_ACCENT,
            )
            self._player.draw(self.surface)

        if self._metrics_panel:
            self._metrics_panel.draw(self.surface)

        self._controls.draw_review(self.surface)

        pygame.display.flip()
        self._clock.tick(config.RENDER_FPS)

        # Drain ALL events once, then distribute
        raw_events = pygame.event.get()

        # Buttons
        button_events = self._controls.handle_review_events(raw_events)

        # Timeline scrub
        if self._player:
            self._player.handle_events(raw_events)

        # Keyboard
        kb_events = self._extract_review_kb_events(raw_events)

        return kb_events + button_events

    def quit(self) -> None:
        pygame.quit()

    # ── Private rendering helpers ──────────────────────────────────────────────

    def _draw_header(self, title: str, status: str, status_color) -> None:
        rect = pygame.Rect(0, 0, config.DISPLAY_WIDTH, config.HEADER_HEIGHT)
        pygame.draw.rect(self.surface, config.COLOR_PANEL, rect)
        pygame.draw.line(
            self.surface, config.COLOR_BORDER,
            (0, config.HEADER_HEIGHT - 1),
            (config.DISPLAY_WIDTH, config.HEADER_HEIGHT - 1), 1
        )

        title_surf = self._fonts["title"].render(title, True, config.COLOR_TEXT)
        self.surface.blit(
            title_surf,
            (20, (config.HEADER_HEIGHT - title_surf.get_height()) // 2)
        )

        if status_color is None:
            status_color = config.COLOR_TEXT_DIM
        status_surf = self._fonts["medium"].render(status, True, status_color)
        sx = config.DISPLAY_WIDTH - status_surf.get_width() - 20
        sy = (config.HEADER_HEIGHT - status_surf.get_height()) // 2

        # Pulsing dot for RECORDING state
        if status_color == config.COLOR_RECORD_DOT:
            alpha    = int(128 + 127 * math.sin(self._record_pulse))
            dot_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (*config.COLOR_RECORD_DOT, alpha), (6, 6), 6)
            self.surface.blit(dot_surf, (sx - 20, sy + 4))

        self.surface.blit(status_surf, (sx, sy))

    def _draw_dual_preview(
        self, frame0: np.ndarray | None, frame2: np.ndarray | None
    ) -> None:
        """Live dual-camera preview used in BUFFERING state."""
        y_start = config.HEADER_HEIGHT
        cell_w  = config.VIDEO_CELL_WIDTH
        cell_h  = config.VIDEO_AREA_HEIGHT

        for i, (frame, label) in enumerate(
            [(frame0, "Face-On"), (frame2, "Down-the-Line")]
        ):
            cell_x    = i * cell_w
            cell_rect = pygame.Rect(cell_x, y_start, cell_w, cell_h)
            pygame.draw.rect(self.surface, config.COLOR_PANEL, cell_rect)

            if frame is not None:
                pg_surf = _bgr_to_pygame(frame, config.VIDEO_DISPLAY_WIDTH, config.VIDEO_DISPLAY_HEIGHT)
                fx = cell_x + (cell_w - config.VIDEO_DISPLAY_WIDTH)  // 2
                fy = y_start + (cell_h - config.VIDEO_DISPLAY_HEIGHT) // 2
                self.surface.blit(pg_surf, (fx, fy))
            else:
                no_cam = self._fonts["medium"].render(
                    f"No {label} Camera", True, config.COLOR_TEXT_DIM
                )
                self.surface.blit(
                    no_cam,
                    (cell_x + (cell_w - no_cam.get_width()) // 2,
                     y_start + cell_h // 2),
                )

            lbl = self._fonts["small"].render(
                f"CAM {i + 1} — {label}", True, config.COLOR_TEXT_DIM
            )
            self.surface.blit(lbl, (cell_x + 10, y_start + 8))

        pygame.draw.line(
            self.surface, config.COLOR_BORDER,
            (cell_w, y_start), (cell_w, y_start + cell_h), 1
        )

    def _draw_buffering_instructions(self, audio_available: bool) -> None:
        y = config.HEADER_HEIGHT + config.VIDEO_AREA_HEIGHT + 15
        if audio_available:
            msg = "Hit the ball when ready   |   SPACE to trigger manually"
        else:
            msg = "No audio device detected   |   Press SPACE to trigger manually"
        surf = self._fonts["small"].render(msg, True, config.COLOR_TEXT_DIM)
        x    = (config.DISPLAY_WIDTH - surf.get_width()) // 2
        self.surface.blit(surf, (x, y))

        if audio_available:
            bar_w  = 300
            bar_h  = 8
            bx     = (config.DISPLAY_WIDTH - bar_w) // 2
            by     = y + surf.get_height() + 8
            pygame.draw.rect(self.surface, config.COLOR_BORDER,
                             (bx, by, bar_w, bar_h), border_radius=4)

    def _draw_analyzing_overlay(self, progress: float) -> None:
        cx = config.DISPLAY_WIDTH  // 2
        cy = config.DISPLAY_HEIGHT // 2

        msg = self._fonts["large"].render("Analyzing your swing…", True, config.COLOR_TEXT)
        self.surface.blit(msg, (cx - msg.get_width() // 2, cy - 80))

        bar_w  = 600
        bar_h  = 12
        bx     = cx - bar_w // 2
        by     = cy
        pygame.draw.rect(self.surface, config.COLOR_BORDER, (bx, by, bar_w, bar_h), border_radius=6)
        fill_w = int(bar_w * max(0, min(1, progress)))
        if fill_w > 0:
            pygame.draw.rect(self.surface, config.COLOR_ACCENT, (bx, by, fill_w, bar_h), border_radius=6)

        pct = self._fonts["medium"].render(
            f"{int(progress * 100)}%", True, config.COLOR_TEXT_DIM
        )
        self.surface.blit(pct, (cx - pct.get_width() // 2, by + bar_h + 12))

        sub = self._fonts["small"].render(
            "This takes about 30–60 seconds. Step back and get ready for your next shot.",
            True, config.COLOR_TEXT_DIM,
        )
        self.surface.blit(sub, (cx - sub.get_width() // 2, cy + 80))

    def _draw_countdown_overlay(self, seconds_remaining: float) -> None:
        """Large countdown number centred on a semi-transparent dark overlay."""
        # Use the pre-allocated overlay (set_alpha) instead of a per-frame
        # SRCALPHA surface — the SRCALPHA path is slow/broken on Pi display drivers
        self.surface.blit(self._countdown_overlay, (0, 0))

        cx = config.DISPLAY_WIDTH  // 2
        cy = config.DISPLAY_HEIGHT // 2

        if seconds_remaining > 0:
            label = str(math.ceil(seconds_remaining))
            color = config.COLOR_TEXT
        else:
            label = "GO!"
            color = config.COLOR_GOOD

        num_surf = self._fonts["huge"].render(label, True, color)
        self.surface.blit(
            num_surf,
            (cx - num_surf.get_width() // 2, cy - num_surf.get_height() // 2),
        )

        hint = self._fonts["large"].render(
            "Walk up, set up, and swing", True, config.COLOR_TEXT_DIM
        )
        self.surface.blit(
            hint,
            (cx - hint.get_width() // 2,
             cy + num_surf.get_height() // 2 + 24),
        )

    def _draw_manual_recording_overlay(self, elapsed: float, total: float) -> None:
        """'Recording… Xs remaining' progress display for manual capture."""
        cx = config.DISPLAY_WIDTH  // 2
        cy = config.DISPLAY_HEIGHT // 2

        # Pulsing REC dot
        alpha    = int(128 + 127 * math.sin(self._record_pulse))
        dot_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.circle(dot_surf, (*config.COLOR_RECORD_DOT, alpha), (12, 12), 12)

        rec_surf  = self._fonts["large"].render("Recording…", True, config.COLOR_TEXT)
        dot_x     = cx - rec_surf.get_width() // 2 - 36
        dot_y     = cy - 44
        self.surface.blit(dot_surf, (dot_x, dot_y))
        self.surface.blit(rec_surf, (cx - rec_surf.get_width() // 2, cy - 40))

        remaining = max(0.0, total - elapsed)
        time_surf = self._fonts["large"].render(
            f"{remaining:.1f}s remaining", True, config.COLOR_TEXT_DIM
        )
        self.surface.blit(time_surf, (cx - time_surf.get_width() // 2, cy + 20))

        # Progress bar
        bar_w = 600
        bar_h = 12
        bx    = cx - bar_w // 2
        by    = cy + 80
        pygame.draw.rect(
            self.surface, config.COLOR_BORDER, (bx, by, bar_w, bar_h), border_radius=6
        )
        fill_w = int(bar_w * min(1.0, elapsed / total)) if total > 0 else 0
        if fill_w > 0:
            pygame.draw.rect(
                self.surface, config.COLOR_RECORD_DOT,
                (bx, by, fill_w, bar_h), border_radius=6
            )

    def _drain_events(self) -> list[str]:
        """Drain all pygame events and return named event strings.
        Used in all states except REVIEW (which has its own drain loop).
        """
        events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events.append("QUIT")
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE: events.append("QUIT")
                elif event.key == pygame.K_SPACE:  events.append("SPACE")
                elif event.key == pygame.K_r:      events.append("REPLAY")
                elif event.key == pygame.K_n:      events.append("NEW_SWING")
                elif event.key == pygame.K_s:      events.append("SAVE")
        return events

    def _extract_review_kb_events(self, raw_events: list) -> list[str]:
        """Map raw pygame events to review-mode named event strings.
        Called after pygame.event.get() has already been called for this frame.
        """
        events = []
        for event in raw_events:
            if event.type == pygame.QUIT:
                events.append("QUIT")
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE: events.append("QUIT")
                elif event.key == pygame.K_SPACE:  events.append("PLAY_PAUSE")
                elif event.key == pygame.K_LEFT:   events.append("STEP_BACK")
                elif event.key == pygame.K_RIGHT:  events.append("STEP_FORWARD")
                elif event.key == pygame.K_r:      events.append("REPLAY")
                elif event.key == pygame.K_n:      events.append("NEW_SWING")
                elif event.key == pygame.K_s:      events.append("SAVE")
        return events


# ── Module-level helpers ───────────────────────────────────────────────────────

def _load_fonts() -> dict:
    pygame.font.init()
    try:
        def f(size): return pygame.font.SysFont("liberationsans", size)
    except Exception:
        def f(size): return pygame.font.Font(None, size)
    return {
        "huge":   f(180),
        "title":  f(28),
        "large":  f(36),
        "medium": f(22),
        "small":  f(18),
        "score":  f(32),
        "metric": f(20),
    }


def _bgr_to_pygame(
    bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> pygame.Surface:
    """Convert a BGR numpy frame to a scaled pygame Surface."""
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # np.transpose returns a non-contiguous view; make_surface on older pygame/Pi
    # requires a C-contiguous array, so force contiguous copy first.
    arr = np.ascontiguousarray(np.transpose(resized, (1, 0, 2)))
    return pygame.surfarray.make_surface(arr)
