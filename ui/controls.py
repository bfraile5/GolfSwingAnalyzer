"""Control bar buttons for the review and buffering screens."""
from __future__ import annotations
import pygame

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class Button:
    def __init__(
        self, label: str, event_name: str,
        rect: pygame.Rect, fonts: dict,
    ) -> None:
        self.label = label
        self.event_name = event_name
        self.rect = rect
        self._fonts = fonts
        self._hovered = False

    def draw(self, surface: pygame.Surface) -> None:
        color = config.COLOR_BUTTON_HOVER if self._hovered else config.COLOR_BUTTON_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, config.COLOR_BORDER, self.rect, 1, border_radius=6)
        text = self._fonts["medium"].render(self.label, True, config.COLOR_TEXT)
        tx = self.rect.x + (self.rect.width - text.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text.get_height()) // 2
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
        self._surface = surface
        self._fonts = fonts
        self._review_buttons = self._make_review_buttons(fonts)

    def draw_buffering(self) -> None:
        """Minimal bar during buffering — just shows keybind hints."""
        y = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        pygame.draw.rect(
            self._surface, config.COLOR_PANEL,
            (0, y, config.DISPLAY_WIDTH, config.CONTROLS_HEIGHT)
        )
        pygame.draw.line(
            self._surface, config.COLOR_BORDER,
            (0, y), (config.DISPLAY_WIDTH, y), 1
        )
        hint = self._fonts["small"].render(
            "SPACE — Trigger  |  ESC — Quit",
            True, config.COLOR_TEXT_DIM
        )
        hx = (config.DISPLAY_WIDTH - hint.get_width()) // 2
        hy = y + (config.CONTROLS_HEIGHT - hint.get_height()) // 2
        self._surface.blit(hint, (hx, hy))

    def draw_review(self, surface: pygame.Surface) -> list[str]:
        """Draw review buttons. Returns list of fired event names."""
        y = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        pygame.draw.rect(
            surface, config.COLOR_PANEL,
            (0, y, config.DISPLAY_WIDTH, config.CONTROLS_HEIGHT)
        )
        pygame.draw.line(surface, config.COLOR_BORDER, (0, y), (config.DISPLAY_WIDTH, y), 1)

        fired = []
        for btn in self._review_buttons:
            btn.draw(surface)

        # Handle events for buttons
        for event in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            for btn in self._review_buttons:
                result = btn.handle_event(event)
                if result:
                    fired.append(result)
        for event in pygame.event.get(pygame.MOUSEMOTION):
            for btn in self._review_buttons:
                btn.handle_event(event)

        return fired

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_review_buttons(fonts: dict) -> list[Button]:
        y = config.DISPLAY_HEIGHT - config.CONTROLS_HEIGHT
        btn_h = config.CONTROLS_HEIGHT - 12
        btn_y = y + 6
        btn_w = 160

        buttons_spec = [
            ("◄ REPLAY",    "REPLAY",    0),
            ("½x SPEED",    "SPEED_HALF", 1),
            ("1x SPEED",    "SPEED_NORM", 2),
            ("NEW SWING",   "NEW_SWING",  3),
            ("SAVE",        "SAVE",       4),
        ]

        total = len(buttons_spec)
        spacing = (config.DISPLAY_WIDTH - total * btn_w) // (total + 1)
        buttons = []
        for label, event, idx in buttons_spec:
            bx = spacing + idx * (btn_w + spacing)
            rect = pygame.Rect(bx, btn_y, btn_w, btn_h)
            buttons.append(Button(label, event, rect, fonts))
        return buttons
