#!/usr/bin/env python3
"""Golf Swing Analyzer — entry point."""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

# Set SDL to use the first available display (needed for Pi without desktop env)
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("DISPLAY", ":0")

from app import App


def main() -> None:
    app = App()
    try:
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
