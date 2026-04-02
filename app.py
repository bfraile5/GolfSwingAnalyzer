"""Top-level App class: owns all threads and the state machine."""
from __future__ import annotations
import threading
import queue
import time
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

import config
from capture.buffer import RollingBuffer
from capture.camera_pair import CameraPair
from capture.audio_trigger import AudioTrigger
from analysis.pose_runner import PoseRunner
from analysis.phases import detect_phases
from analysis.metrics import compute_metrics, compute_swing_analysis
from utils.annotations import annotate_clip
from utils.saver import save_swing
from ui.screen import Screen


ANALYSIS_WATCHDOG_SECONDS = 180  # increased for full-frame processing


class AppState(Enum):
    SPLASH     = auto()
    BUFFERING  = auto()
    TRIGGERED  = auto()
    ANALYZING  = auto()
    REVIEW     = auto()


@dataclass
class SwingReport:
    frames_cam0: list
    frames_cam2: list
    phases: object
    metrics: object
    swing_analysis: object = None
    impact_frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)


class App:
    def __init__(self) -> None:
        self._state = AppState.SPLASH
        self._running = True

        # Buffers
        maxlen = int(config.BUFFER_SECONDS * config.TARGET_FPS)
        self._buf0 = RollingBuffer(maxlen)
        self._buf2 = RollingBuffer(maxlen)

        # Trigger event
        self._trigger_event = threading.Event()

        # Camera pair
        self._cam_stop_event = threading.Event()
        self._cameras = CameraPair(self._buf0, self._buf2, self._cam_stop_event)

        # Audio trigger
        self._audio = AudioTrigger(self._trigger_event)
        self._trigger_timestamp: float = 0.0

        # Analysis output queue
        self._results_queue: queue.Queue[SwingReport] = queue.Queue()
        self._analysis_progress = 0.0

        # Current report for review
        self._report: SwingReport | None = None

        # Analysis watchdog
        self._analysis_start_time: float = 0.0

        # Screen (initialised last so pygame starts once everything is ready)
        self._screen = Screen()

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        last_time = time.monotonic()

        # Start cameras
        if not self._cameras.open():
            print(f"[App] Camera error: {self._cameras.error}")

        self._cam_thread = threading.Thread(
            target=self._cameras.run, daemon=True, name="CaptureThread"
        )
        self._cam_thread.start()

        self._audio.start()
        self._state = AppState.BUFFERING

        while self._running:
            now = time.monotonic()
            dt_ms = (now - last_time) * 1000
            last_time = now
            self._tick(dt_ms)

        self._shutdown()

    # ── State machine ──────────────────────────────────────────────────────────

    def _tick(self, dt_ms: float) -> None:
        if self._state == AppState.BUFFERING:
            self._tick_buffering()

        elif self._state == AppState.TRIGGERED:
            if not self._cam_thread.is_alive():
                self._state = AppState.ANALYZING
                self._start_analysis()
            else:
                events = self._screen.render_analyzing(0.0)
                self._handle_global_events(events)

        elif self._state == AppState.ANALYZING:
            self._tick_analyzing()

        elif self._state == AppState.REVIEW:
            events = self._screen.render_review(dt_ms)
            self._handle_review_events(events)

        elif self._state == AppState.SPLASH:
            events = self._screen.render_splash()
            self._handle_global_events(events)

    def _tick_buffering(self) -> None:
        if self._trigger_event.is_set():
            self._trigger_event.clear()
            self._trigger_timestamp = self._audio.trigger_timestamp or time.monotonic()
            self._cam_stop_event.set()
            self._audio.stop()
            self._state = AppState.TRIGGERED
            return

        frame0 = self._buf0.latest_frame()
        frame2 = self._buf2.latest_frame()
        events = self._screen.render_buffering(
            frame0, frame2, self._audio.available
        )
        self._handle_buffering_events(events)

    def _tick_analyzing(self) -> None:
        if (self._analysis_start_time > 0 and
                time.monotonic() - self._analysis_start_time > ANALYSIS_WATCHDOG_SECONDS):
            print("[App] Analysis watchdog triggered — resetting")
            self._analysis_start_time = 0.0
            self._reset_for_new_swing()
            return

        events = self._screen.render_analyzing(self._analysis_progress)
        self._handle_global_events(events)

        try:
            report = self._results_queue.get_nowait()
            self._report = report
            self._screen.load_review(report)
            self._state = AppState.REVIEW
        except queue.Empty:
            pass

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _handle_global_events(self, events: list[str]) -> None:
        for e in events:
            if e == "QUIT":
                self._running = False
            elif e == "SPACE":
                if self._state == AppState.BUFFERING:
                    self._audio.manual_trigger()

    def _handle_buffering_events(self, events: list[str]) -> None:
        self._handle_global_events(events)

    def _handle_review_events(self, events: list[str]) -> None:
        for e in events:
            if e == "QUIT":
                self._running = False
            elif e == "REPLAY":
                if self._screen._player:
                    self._screen._player.reset()
            elif e in ("SPEED_HALF",):
                if self._screen._player:
                    self._screen._player.set_speed(0.5)
            elif e in ("SPEED_NORM",):
                if self._screen._player:
                    self._screen._player.set_speed(1.0)
            elif e == "NEW_SWING":
                self._reset_for_new_swing()
            elif e == "SAVE":
                self._save_current_swing()

    # ── Analysis ───────────────────────────────────────────────────────────────

    def _start_analysis(self) -> None:
        self._analysis_start_time = time.monotonic()
        t = threading.Thread(
            target=self._analysis_worker, daemon=True, name="AnalysisThread"
        )
        t.start()

    def _analysis_worker(self) -> None:
        try:
            self._analysis_progress = 0.05

            clip0 = self._buf0.snapshot()
            clip2 = self._buf2.snapshot()

            # Find impact frame index (frame closest to trigger timestamp)
            impact_frame_idx = 0
            if self._trigger_timestamp and clip0:
                impact_frame_idx = min(
                    range(len(clip0)),
                    key=lambda i: abs(clip0[i][0] - self._trigger_timestamp),
                )

            if not clip0 and not clip2:
                print("[Analysis] No frames captured — aborting")
                self._results_queue.put(self._empty_report())
                return

            self._analysis_progress = 0.10

            # Process cam0 (face-on) — all frames with temporal smoothing
            with PoseRunner() as runner0:
                results0 = runner0.process_clip(clip0)
            self._analysis_progress = 0.45

            # Process cam2 (down-the-line) — all frames with temporal smoothing
            with PoseRunner() as runner2:
                results2 = runner2.process_clip(clip2)
            self._analysis_progress = 0.75

            # Detect P1–P10 positions (pass audio trigger frame to anchor P7)
            phases = detect_phases(results0, impact_frame_idx=impact_frame_idx)

            # Compute scorecard metrics (6 scores + overall)
            metrics = compute_metrics(results0, results2, phases)

            # Compute detailed P1–P10 position analysis
            swing_analysis = compute_swing_analysis(results0, results2, phases)
            self._analysis_progress = 0.85

            # Annotate frames with skeleton overlay and phase labels
            annotated0 = annotate_clip(clip0, results0, phases)
            annotated2 = annotate_clip(clip2, results2, phases)
            self._analysis_progress = 0.97

            report = SwingReport(
                frames_cam0=annotated0,
                frames_cam2=annotated2,
                phases=phases,
                metrics=metrics,
                swing_analysis=swing_analysis,
                impact_frame_idx=impact_frame_idx,
            )
            self._results_queue.put(report)
            self._analysis_progress = 1.0
            print(f"[Analysis] Done — overall score: {metrics.overall_score}")
            print(f"[Analysis] Tempo: {metrics.tempo_ratio:.2f}:1  (BS {metrics.backswing_duration:.2f}s / DS {metrics.downswing_duration:.2f}s)")
            print(f"[Analysis] Address: {swing_analysis.address_evaluation}")
            print(f"[Analysis] Backswing: {swing_analysis.backswing_evaluation}")
            print(f"[Analysis] Transition: {swing_analysis.transition_evaluation}")
            print(f"[Analysis] Impact: {swing_analysis.impact_evaluation}")
            print(f"[Analysis] Follow-through: {swing_analysis.follow_through_evaluation}")

        except Exception as e:
            import traceback
            print(f"[Analysis] Error: {e}")
            traceback.print_exc()
            self._results_queue.put(self._empty_report())

    def _empty_report(self) -> SwingReport:
        from analysis.metrics import SwingMetrics
        from analysis.phases import SwingPhases
        return SwingReport(
            frames_cam0=[],
            frames_cam2=[],
            phases=SwingPhases(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            metrics=SwingMetrics(),
            swing_analysis=None,
        )

    # ── Reset & Save ───────────────────────────────────────────────────────────

    def _reset_for_new_swing(self) -> None:
        self._buf0.clear()
        self._buf2.clear()
        self._trigger_event.clear()
        self._cam_stop_event.clear()
        self._analysis_progress = 0.0
        self._report = None

        self._cameras = CameraPair(self._buf0, self._buf2, self._cam_stop_event)
        if not self._cameras.open():
            print(f"[App] Camera reopen error: {self._cameras.error}")

        self._cam_thread = threading.Thread(
            target=self._cameras.run, daemon=True, name="CaptureThread"
        )
        self._cam_thread.start()

        self._audio.start()
        self._state = AppState.BUFFERING

    def _save_current_swing(self) -> None:
        if self._report is None:
            return
        try:
            path = save_swing(
                self._report.frames_cam0,
                self._report.frames_cam2,
                self._report.metrics,
                self._report.phases,
                self._report.swing_analysis,
            )
            print(f"[App] Swing saved to {path}")
        except Exception as e:
            print(f"[App] Save failed: {e}")

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        self._audio.stop()
        self._cam_stop_event.set()
        self._cameras.release()
        self._screen.quit()
