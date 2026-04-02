"""Metrics panel: six score cards with bars and tip text."""
from __future__ import annotations
import pygame

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from analysis.metrics import SwingMetrics

# Ordered list of (display_name, metrics_attr)
METRIC_DEFS = [
    ("Spine Tilt",      "spine_angle_score"),
    ("Hip Rotation",    "hip_rotation_score"),
    ("Knee Flex",       "knee_flex_score"),
    ("Head Stability",  "head_stability_score"),
    ("Arm Extension",   "arm_extension_score"),
    ("Swing Plane",     "swing_plane_score"),
]


class MetricsPanel:
    """Renders the bottom metrics strip and an overall score badge."""

    def __init__(self, metrics: SwingMetrics, fonts: dict) -> None:
        self.metrics = metrics
        self._fonts = fonts

    def draw(self, surface: pygame.Surface) -> None:
        y = config.HEADER_HEIGHT + config.VIDEO_AREA_HEIGHT
        w = config.DISPLAY_WIDTH
        h = config.METRICS_HEIGHT

        # Panel background
        pygame.draw.rect(surface, config.COLOR_PANEL, (0, y, w, h))
        pygame.draw.line(surface, config.COLOR_BORDER, (0, y), (w, y), 1)

        n = len(METRIC_DEFS)
        # Reserve 400px on right: 190px for tempo + 10px gap + 190px for overall
        card_w = (w - 400) // n
        card_h = h - 20
        card_y = y + 10

        for i, (name, attr) in enumerate(METRIC_DEFS):
            score = getattr(self.metrics, attr, 0.0)
            tip = self.metrics.tips.get(name, "")
            cx = i * card_w + 10
            self._draw_card(surface, cx, card_y, card_w - 10, card_h, name, score, tip)

        # Tempo box
        self._draw_tempo(surface, w - 400, card_y, 190, card_h)

        # Overall score on the right
        self._draw_overall(surface, w - 200, card_y, 190, card_h)

    # ── Private ────────────────────────────────────────────────────────────────

    def _draw_card(
        self,
        surface: pygame.Surface,
        x: int, y: int, w: int, h: int,
        name: str, score: float, tip: str,
    ) -> None:
        # Card border
        pygame.draw.rect(surface, config.COLOR_BORDER, (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (30, 30, 55), (x + 1, y + 1, w - 2, h - 2), border_radius=6)

        inner_x = x + 8
        inner_w = w - 16
        iy = y + 8

        # Metric name
        name_surf = self._fonts["small"].render(name, True, config.COLOR_TEXT_DIM)
        surface.blit(name_surf, (inner_x, iy))
        iy += name_surf.get_height() + 4

        # Score number
        score_color = _score_color(score)
        score_surf = self._fonts["score"].render(str(int(score)), True, score_color)
        surface.blit(score_surf, (inner_x, iy))
        iy += score_surf.get_height() + 4

        # Score bar
        bar_h = 6
        bar_w = inner_w
        pygame.draw.rect(surface, config.COLOR_BG, (inner_x, iy, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * score / 100)
        if fill_w > 0:
            pygame.draw.rect(surface, score_color, (inner_x, iy, fill_w, bar_h), border_radius=3)
        iy += bar_h + 6

        # Tip text (clipped to card width)
        if tip:
            tip_surf = self._fonts["small"].render(tip, True, config.COLOR_TEXT_DIM)
            # Clip to card width by subsurface if needed
            if tip_surf.get_width() > inner_w:
                tip_surf = tip_surf.subsurface((0, 0, inner_w, tip_surf.get_height()))
            surface.blit(tip_surf, (inner_x, iy))

    def _draw_tempo(
        self, surface: pygame.Surface,
        x: int, y: int, w: int, h: int,
    ) -> None:
        ratio = self.metrics.tempo_ratio
        bs = self.metrics.backswing_duration
        ds = self.metrics.downswing_duration

        # Color based on proximity to 3.0 target
        if config.TEMPO_GREEN_MIN <= ratio <= config.TEMPO_GREEN_MAX:
            color = config.COLOR_GOOD
        elif config.TEMPO_AMBER_MIN <= ratio <= config.TEMPO_AMBER_MAX:
            color = config.COLOR_WARN
        elif ratio > 0:
            color = config.COLOR_BAD
        else:
            color = config.COLOR_TEXT_DIM

        pygame.draw.rect(surface, config.COLOR_BORDER, (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (30, 30, 55), (x + 1, y + 1, w - 2, h - 2), border_radius=8)

        label = self._fonts["small"].render("TEMPO", True, config.COLOR_TEXT_DIM)
        surface.blit(label, (x + (w - label.get_width()) // 2, y + 8))

        if ratio > 0:
            ratio_str = f"{ratio:.1f}:1"
        else:
            ratio_str = "—"
        big = self._fonts["score"].render(ratio_str, True, color)
        surface.blit(big, (x + (w - big.get_width()) // 2, y + 30))

        # Backswing / downswing times
        iy = y + 30 + big.get_height() + 4
        if ratio > 0:
            bs_surf = self._fonts["small"].render(f"BS {bs:.2f}s", True, config.COLOR_TEXT_DIM)
            ds_surf = self._fonts["small"].render(f"DS {ds:.2f}s", True, config.COLOR_TEXT_DIM)
            surface.blit(bs_surf, (x + (w - bs_surf.get_width()) // 2, iy))
            surface.blit(ds_surf, (x + (w - ds_surf.get_width()) // 2, iy + bs_surf.get_height() + 2))
            iy += bs_surf.get_height() * 2 + 6

        target_surf = self._fonts["small"].render(f"Target 3.0:1", True, config.COLOR_TEXT_DIM)
        surface.blit(target_surf, (x + (w - target_surf.get_width()) // 2, iy))

    def _draw_overall(
        self, surface: pygame.Surface,
        x: int, y: int, w: int, h: int,
    ) -> None:
        score = self.metrics.overall_score
        score_color = _score_color(score)

        pygame.draw.rect(surface, config.COLOR_BORDER, (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (30, 30, 55), (x + 1, y + 1, w - 2, h - 2), border_radius=8)

        label = self._fonts["small"].render("OVERALL", True, config.COLOR_TEXT_DIM)
        surface.blit(label, (x + (w - label.get_width()) // 2, y + 10))

        big = self._fonts["large"].render(str(int(score)), True, score_color)
        surface.blit(big, (x + (w - big.get_width()) // 2, y + h // 2 - big.get_height() // 2))

        grade = _grade(score)
        grade_surf = self._fonts["medium"].render(grade, True, score_color)
        surface.blit(grade_surf, (x + (w - grade_surf.get_width()) // 2, y + h - 30))


def _score_color(score: float) -> tuple:
    if score >= config.SCORE_GREEN_THRESHOLD:
        return config.COLOR_GOOD
    if score >= config.SCORE_AMBER_THRESHOLD:
        return config.COLOR_WARN
    return config.COLOR_BAD


def _grade(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 80: return "Good"
    if score >= 70: return "Fair"
    if score >= 60: return "Needs Work"
    return "Keep Practicing"
