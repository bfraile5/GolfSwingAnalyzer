"""Thread-safe circular frame buffer using JPEG compression to minimise RAM."""
import threading
import collections
import time
import cv2
import numpy as np

JPEG_QUALITY = 85


class RollingBuffer:
    """Stores (timestamp, jpeg_bytes) pairs in a fixed-length deque.

    Frames are JPEG-compressed on append so the buffer stays small (~24 MB
    for 10 s at 30 FPS from a 640×480 camera).
    """

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._deque: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def append(self, frame: np.ndarray) -> None:
        ts = time.monotonic()
        frame = np.ascontiguousarray(frame)
        ok, buf = cv2.imencode(".jpg", frame, self._encode_params)
        if ok:
            with self._lock:
                self._deque.append((ts, buf.tobytes()))

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        """Return a list of (timestamp, decoded_bgr_frame) for all buffered frames."""
        with self._lock:
            items = list(self._deque)
        # Decode outside the lock so we don't hold it during heavy work
        result = []
        for ts, jpeg_bytes in items:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                result.append((ts, frame))
        return result

    def latest_frame(self) -> np.ndarray | None:
        """Return most recent decoded frame for live preview (no lock held long)."""
        with self._lock:
            if not self._deque:
                return None
            ts, jpeg_bytes = self._deque[-1]
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)
