#!/bin/bash
# Golf Swing Analyzer — launch script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $SCRIPT_DIR/.venv"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

# Pre-flight check
echo "Running hardware check..."
"$VENV" check_hardware.py
if [ $? -ne 0 ]; then
    echo "Hardware check failed. Fix issues above before launching."
    exit 1
fi

echo "Starting Golf Swing Analyzer..."
exec "$VENV" main.py "$@"
