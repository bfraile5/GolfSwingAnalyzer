"""Dual-camera capture thread feeding two RollingBuffers."""
import threading
import time
import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from capture.buffer import RollingBuffer


class CameraPair:
    """Opens two cameras and feeds frames into two RollingBuffers.

    After stop_event is set (by the trigger), the thread continues capturing
    for POST_TRIGGER_SECONDS more seconds before exiting so we get the
    follow-through.
    """

    def __init__(
        self,
        buf0: RollingBuffer,
        buf2: RollingBuffer,
        stop_event: threading.Event,
    ) -> None:
        self.buf0 = buf0
        self.buf2 = buf2
        self.stop_event = stop_event
        self._immediate_stop = threading.Event()
        self.cap0: cv2.VideoCapture | None = None
        self.cap2: cv2.VideoCapture | None = None
        self._error: str | None = None
        self._post_trigger_frames = int(
            config.POST_TRIGGER_SECONDS * config.TARGET_FPS
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open both cameras. Returns True on success."""
        self.cap0 = self._open_cam(config.CAM0_INDEX)
        if self.cap0 is None:
            self._error = f"Cannot open camera 0 (index {config.CAM0_INDEX})"
            return False
        self.cap2 = self._open_cam(config.CAM2_INDEX)
        if self.cap2 is None:
            self._error = f"Cannot open camera 2 (index {config.CAM2_INDEX})"
            self.cap0.release()
            return False
        return True

    def run(self) -> None:
        """Main capture loop — run in a dedicated thread."""
        if self.cap0 is None or self.cap2 is None:
            return

        frame_interval = 1.0 / config.TARGET_FPS
        post_frames_remaining = self._post_trigger_frames
        triggered = False

        while True:
            t_start = time.monotonic()

            # Grab both cameras near-simultaneously, then retrieve
            grabbed0 = self.cap0.grab()
            grabbed2 = self.cap2.grab()

            if not grabbed0 or not grabbed2:
                time.sleep(0.05)
                continue

            ok0, frame0 = self.cap0.retrieve()
            ok2, frame2 = self.cap2.retrieve()

            if ok0 and frame0 is not None:
                self.buf0.append(frame0)
            if ok2 and frame2 is not None:
                self.buf2.append(frame2)

            # Immediate stop (manual-trigger countdown path)
            if self._immediate_stop.is_set():
                break

            # Handle post-trigger countdown
            if self.stop_event.is_set():
                if not triggered:
                    triggered = True
                post_frames_remaining -= 1
                if post_frames_remaining <= 0:
                    break

            # Pace to TARGET_FPS
            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def immediate_stop(self) -> None:
        """Signal run() to exit on the next frame, without post-trigger delay."""
        self._immediate_stop.set()

    def record_fixed(
        self, n_seconds: float
    ) -> tuple[list[tuple[float, np.ndarray]], list[tuple[float, np.ndarray]]]:
        """Capture exactly n_seconds of frames from the already-open cameras.

        Returns (frames_cam0, frames_cam2) as lists of (timestamp, bgr_frame).
        Must be called after run() has exited (i.e. after immediate_stop() +
        thread join) so both threads are not competing for the same caps.
        """
        if self.cap0 is None or self.cap2 is None:
            return [], []

        frames0: list[tuple[float, np.ndarray]] = []
        frames2: list[tuple[float, np.ndarray]] = []
        frame_interval = 1.0 / config.TARGET_FPS
        end_time = time.monotonic() + n_seconds

        while time.monotonic() < end_time:
            t_start = time.monotonic()

            grabbed0 = self.cap0.grab()
            grabbed2 = self.cap2.grab()

            if grabbed0 and grabbed2:
                ok0, frame0 = self.cap0.retrieve()
                ok2, frame2 = self.cap2.retrieve()
                ts = time.monotonic()
                if ok0 and frame0 is not None:
                    frames0.append((ts, frame0))
                if ok2 and frame2 is not None:
                    frames2.append((ts, frame2))

            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[CameraPair] record_fixed: captured {len(frames0)} / {len(frames2)} frames")
        return frames0, frames2

    def release(self) -> None:
        if self.cap0:
            self.cap0.release()
        if self.cap2:
            self.cap2.release()

    @property
    def error(self) -> str | None:
        return self._error

    # ── Private helpers ────────────────────────────────────────────────────────

    def _open_cam(self, index: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency
        return cap
