"""Control bar buttons for the review and buffering screens."""
from __future__ import annotations
import pygame

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class Button:
    def __init__(
        self,
        label: str,
        event_name: str,
        rect: pygame.Rect,
        fonts: dict,
        accent: bool = False,
    ) -> None:
        self.label      = label
        self.event_name = event_name
        self.rect       = rect
        self._fonts     = fonts
        self._accent    = accent
        self._hovered   = False

    def draw(self, surface: pygame.Surface) -> None:
        if self._accent:
            bg     = (45, 80, 45) if not self._hovered else (65, 110, 65)
            border = config.COLOR_GOOD
        else:
            bg     = config.COLOR_BUTTON_HOVER if self._hovered else config.COLOR_BUTTON_BG
            border = config.COLOR_BORDER
        pygame.draw.rect(surface, bg,     self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=8)
        text = self._fonts["medium"].render(self.label, True, config.COLOR_TEXT)
        tx   = self.rect.x + (self.rect.width  - text.get_width())  // 2
        ty   = self.rect.y + (self.rect.height - text.get_height()) // 2
        surface.blit(text, (tx, ty))

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.event_name
        return None


class ControlBar:
    """Renders the bottom control bar."""

    def __init__(self, surface: pygame.Surface, fonts: dict) -> None:
        self._surface        = surface
        self._fonts          = fonts
        self._review_buttons = self._make_review_buttons(fonts)

    # ── Buffering mode ─────────────────────────────────────────────────────────

    def draw_buffering(self) -> None:
        """Minimal hint bar shown while waiting for a swing."""
        y = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        pygame.draw.rect(
            self._surface, config.COLOR_PANEL,
            (0, y, config.DISPLAY_WIDTH, config.CONTROLS_HEIGHT),
        )
        pygame.draw.line(
            self._surface, config.COLOR_BORDER,
            (0, y), (config.DISPLAY_WIDTH, y), 1,
        )
        hint = self._fonts["small"].render(
            "SPACE — Trigger  |  ESC — Quit",
            True, config.COLOR_TEXT_DIM,
        )
        hx = (config.DISPLAY_WIDTH  - hint.get_width())  // 2
        hy = y + (config.CONTROLS_HEIGHT - hint.get_height()) // 2
        self._surface.blit(hint, (hx, hy))

    # ── Review mode ────────────────────────────────────────────────────────────

    def draw_review(self, surface: pygame.Surface) -> None:
        """Draw the review control bar (drawing only — no event processing)."""
        y = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        pygame.draw.rect(
            surface, config.COLOR_PANEL,
            (0, y, config.DISPLAY_WIDTH, config.CONTROLS_HEIGHT),
        )
        pygame.draw.line(surface, config.COLOR_BORDER,
                         (0, y), (config.DISPLAY_WIDTH, y), 1)
        for btn in self._review_buttons:
            btn.draw(surface)

    def handle_review_events(self, events: list) -> list[str]:
        """Process pre-drained events against review buttons.
        Returns list of fired event name strings.
        """
        fired = []
        for event in events:
            for btn in self._review_buttons:
                result = btn.handle_event(event)
                if result:
                    fired.append(result)
        return fired

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_review_buttons(fonts: dict) -> list[Button]:
        y     = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        btn_h = config.CONTROLS_HEIGHT - 14
        btn_y = y + 7

        # (label, event_name, width_px, accent)
        specs = [
            ("◄◄",        "STEP_BACK",    80,  False),
            ("▶ / ❚❚",    "PLAY_PAUSE",  110,  False),
            ("►►",        "STEP_FORWARD",  80,  False),
            ("½x",        "SPEED_HALF",    80,  False),
            ("1x",        "SPEED_NORM",    80,  False),
            ("↺ REPLAY",  "REPLAY",       130,  False),
            ("NEW SWING", "NEW_SWING",    155,  False),
            ("✓ SAVE",    "SAVE",         125,  True),
        ]

        total_w  = sum(w for _, _, w, _ in specs)
        n_btns   = len(specs)
        spacing  = (config.DISPLAY_WIDTH - total_w) // (n_btns + 1)

        buttons = []
        x = spacing
        for label, event, w, accent in specs:
            rect = pygame.Rect(x, btn_y, w, btn_h)
            buttons.append(Button(label, event, rect, fonts, accent=accent))
            x += w + spacing
        return buttons
