"""Audio-based swing trigger using sounddevice RMS detection."""
import threading
import time
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False


class AudioTrigger:
    """Listens on the microphone and fires trigger_event on a loud impact.

    Falls back to manual-only mode if sounddevice is unavailable or no
    audio device is found.
    """

    def __init__(self, trigger_event: threading.Event) -> None:
        self.trigger_event = trigger_event
        self._threshold = config.TRIGGER_RMS_THRESHOLD
        self._cooldown = config.TRIGGER_COOLDOWN_S
        self._last_trigger = 0.0
        self._stream = None
        self._active = False
        self.available = _SD_AVAILABLE

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the audio stream. Returns True if started successfully."""
        if not _SD_AVAILABLE:
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
                blocksize=config.AUDIO_BLOCKSIZE,
                dtype="float32",
                device=config.AUDIO_DEVICE_INDEX,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._active = True
            return True
        except Exception as e:
            print(f"[AudioTrigger] Could not open audio stream: {e}")
            self.available = False
            return False

    def stop(self) -> None:
        self._active = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def manual_trigger(self) -> None:
        """Programmatically fire the trigger (SPACE key fallback)."""
        self._fire()

    def set_threshold(self, value: float) -> None:
        self._threshold = max(0.01, min(1.0, value))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — must be very fast, no allocations."""
        if not self._active:
            return
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms >= self._threshold:
            now = time.monotonic()
            if now - self._last_trigger >= self._cooldown:
                self._last_trigger = now
                self._fire()

    def _fire(self) -> None:
        self._last_trigger = time.monotonic()
        self.trigger_event.set()
