#!/usr/bin/env python3
"""Hardware pre-flight check. Run before launching the main app."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

errors = []
warnings = []

# ── Python version ─────────────────────────────────────────────────────────────
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    errors.append(f"Python 3.10+ required, got {major}.{minor}")
else:
    print(f"[OK] Python {major}.{minor}")

# ── Required packages ──────────────────────────────────────────────────────────
for pkg, import_name in [
    ("opencv-contrib-python", "cv2"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("pygame", "pygame"),
    ("sounddevice", "sounddevice"),
]:
    try:
        mod = __import__(import_name)
        print(f"[OK] {pkg} {getattr(mod, '__version__', '?')}")
    except ImportError:
        errors.append(f"Missing package: {pkg}")

# ── Cameras ────────────────────────────────────────────────────────────────────
try:
    import cv2
    import config
    for idx, name in [(config.CAM0_INDEX, "Face-On"), (config.CAM2_INDEX, "Down-the-Line")]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[OK] Camera {idx} ({name}): {frame.shape[1]}x{frame.shape[0]}")
            else:
                warnings.append(f"Camera {idx} ({name}): opens but cannot read frames")
            cap.release()
        else:
            errors.append(f"Camera {idx} ({name}): cannot open (check index in config.py)")
except Exception as e:
    errors.append(f"Camera check failed: {e}")

# ── Audio ──────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    if default_input >= 0:
        dev = sd.query_devices(default_input)
        print(f"[OK] Audio input: {dev[name]} ({dev[max_input_channels]} ch)")
    else:
        warnings.append("No default audio input device — manual trigger (SPACE) will be used")
except Exception as e:
    warnings.append(f"Audio check failed: {e} — manual trigger only")

# ── Display ────────────────────────────────────────────────────────────────────
display_env = os.environ.get("DISPLAY", "")
if not display_env:
    warnings.append("DISPLAY not set — app may not launch (try: export DISPLAY=:0)")
else:
    print(f"[OK] DISPLAY={display_env}")

# ── Save directory ─────────────────────────────────────────────────────────────
try:
    import config
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    test_file = os.path.join(config.SAVE_DIR, ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print(f"[OK] Save directory: {config.SAVE_DIR}")
except Exception as e:
    warnings.append(f"Save directory issue: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
if warnings:
    for w in warnings:
        print(f"[WARN] {w}")
if errors:
    for e in errors:
        print(f"[FAIL] {e}")
    print(f"\n{len(errors)} error(s) found. Fix before launching.")
    sys.exit(1)
else:
    print("Hardware check passed. Ready to launch.")
    sys.exit(0)
