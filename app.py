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
from analysis.pose_runner import PoseRunner, fuse_dual_camera_poses
from analysis.phases import detect_phases
from analysis.metrics import compute_metrics, compute_swing_analysis
from utils.annotations import annotate_clip
from utils.saver import save_swing
from ui.screen import Screen


ANALYSIS_WATCHDOG_SECONDS = 180  # increased for full-frame processing


class AppState(Enum):
    SPLASH           = auto()
    BUFFERING        = auto()
    COUNTDOWN        = auto()   # spacebar pressed — counting down before recording
    MANUAL_RECORDING = auto()   # 10-second fixed capture after countdown
    TRIGGERED        = auto()
    ANALYZING        = auto()
    REVIEW           = auto()


@dataclass
class SwingReport:
    frames_cam0: list
    frames_cam2: list
    phases: object
    metrics: object
    swing_analysis: object = None
    impact_frame_idx: int = 0
    actual_fps: float = 0.0   # measured capture fps; 0.0 = use config.TARGET_FPS
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

        # Manual-trigger (spacebar) state
        self._countdown_start: float = 0.0
        self._manual_record_start: float = 0.0
        self._manual_record_done = threading.Event()
        self._manual_frames_cam0: list = []
        self._manual_frames_cam2: list = []
        self._manual_actual_fps: float = float(config.TARGET_FPS)

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
            try:
                self._tick(dt_ms)
            except Exception as exc:
                import traceback
                print(f"[App] Tick error in state {self._state}: {exc}")
                traceback.print_exc()
                # Don't crash the whole app on a transient render error;
                # reset to BUFFERING so the user can try again.
                self._reset_for_new_swing()

        self._shutdown()

    # ── State machine ──────────────────────────────────────────────────────────

    def _tick(self, dt_ms: float) -> None:
        if self._state == AppState.BUFFERING:
            self._tick_buffering()

        elif self._state == AppState.COUNTDOWN:
            self._tick_countdown()

        elif self._state == AppState.MANUAL_RECORDING:
            self._tick_manual_recording()

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

    def _start_countdown(self) -> None:
        """Begin the manual-trigger countdown (called on SPACE in BUFFERING)."""
        self._countdown_start = time.monotonic()
        self._state = AppState.COUNTDOWN
        print(f"[App] Countdown started — {config.MANUAL_COUNTDOWN_SECONDS}s")

    def _tick_countdown(self) -> None:
        elapsed    = time.monotonic() - self._countdown_start
        remaining  = config.MANUAL_COUNTDOWN_SECONDS - elapsed
        frame0     = self._buf0.latest_frame()
        frame2     = self._buf2.latest_frame()
        events     = self._screen.render_countdown(remaining, frame0, frame2)
        self._handle_global_events(events)

        # Re-sample elapsed AFTER render so the transition fires at the correct
        # wall-clock time even if a frame took unusually long to draw (e.g. on Pi).
        # Hold "GO!" on screen for one extra second before switching.
        if time.monotonic() - self._countdown_start >= config.MANUAL_COUNTDOWN_SECONDS + 0.5:
            self._start_manual_recording()

    def _start_manual_recording(self) -> None:
        """Stop the rolling-buffer cam thread and begin 10-second fixed capture."""
        print("[App] Starting manual recording")
        self._audio.stop()
        self._cameras.immediate_stop()
        self._cam_thread.join(timeout=2.0)   # wait for run() to exit

        self._manual_record_done.clear()
        self._manual_frames_cam0 = []
        self._manual_frames_cam2 = []
        self._manual_record_start = time.monotonic()

        t = threading.Thread(
            target=self._manual_record_worker, daemon=True, name="ManualRecordThread"
        )
        t.start()
        self._state = AppState.MANUAL_RECORDING

    def _tick_manual_recording(self) -> None:
        elapsed = time.monotonic() - self._manual_record_start
        events  = self._screen.render_manual_recording(elapsed, config.MANUAL_RECORD_SECONDS)
        self._handle_global_events(events)

        if self._manual_record_done.is_set():
            print(f"[App] Manual recording done — {len(self._manual_frames_cam0)} frames")
            self._analysis_start_time = time.monotonic()
            self._analysis_progress   = 0.0
            t = threading.Thread(
                target=self._manual_analysis_worker, daemon=True, name="AnalysisThread"
            )
            t.start()
            self._state = AppState.ANALYZING

    def _manual_record_worker(self) -> None:
        try:
            frames0, frames2, actual_fps = self._cameras.record_fixed(config.MANUAL_RECORD_SECONDS)
            self._manual_frames_cam0 = frames0
            self._manual_frames_cam2 = frames2
            self._manual_actual_fps = actual_fps
        except Exception as e:
            import traceback
            print(f"[ManualRecord] Error: {e}")
            traceback.print_exc()
            self._manual_frames_cam0 = []
            self._manual_frames_cam2 = []
        finally:
            self._manual_record_done.set()

    def _manual_analysis_worker(self) -> None:
        """Analysis worker for manually-triggered swings (no audio impact anchor)."""
        try:
            clip0 = self._manual_frames_cam0
            clip2 = self._manual_frames_cam2
            # No audio trigger — let detect_phases find impact from pose data
            impact_frame_idx = 0
            report = self._run_analysis_pipeline(
                clip0, clip2, impact_frame_idx, self._manual_actual_fps
            )
            self._results_queue.put(report)
        except Exception as e:
            import traceback
            print(f"[ManualAnalysis] Error: {e}")
            traceback.print_exc()
            self._results_queue.put(self._empty_report())

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _handle_global_events(self, events: list[str]) -> None:
        for e in events:
            if e == "QUIT":
                self._running = False
            elif e == "SPACE":
                if self._state == AppState.BUFFERING:
                    self._start_countdown()

    def _handle_buffering_events(self, events: list[str]) -> None:
        self._handle_global_events(events)

    def _handle_review_events(self, events: list[str]) -> None:
        for e in events:
            if e == "QUIT":
                self._running = False
            elif e == "REPLAY":
                if self._screen._player:
                    self._screen._player.reset()
            elif e == "PLAY_PAUSE":
                if self._screen._player:
                    self._screen._player.toggle_play()
            elif e == "STEP_FORWARD":
                if self._screen._player:
                    self._screen._player.step_forward()
            elif e == "STEP_BACK":
                if self._screen._player:
                    self._screen._player.step_back()
            elif e == "SPEED_025":
                if self._screen._player:
                    self._screen._player.set_speed(0.25)
            elif e == "SPEED_050":
                if self._screen._player:
                    self._screen._player.set_speed(0.5)
            elif e == "SPEED_100":
                if self._screen._player:
                    self._screen._player.set_speed(1.0)
            elif e == "SPEED_200":
                if self._screen._player:
                    self._screen._player.set_speed(2.0)
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
        """Analysis worker for audio-triggered swings."""
        try:
            self._analysis_progress = 0.05

            clip0 = self._buf0.snapshot()
            clip2 = self._buf2.snapshot()

            # Measure actual capture fps from buffer timestamps
            if len(clip0) > 1:
                buf_duration = clip0[-1][0] - clip0[0][0]
                actual_fps = (len(clip0) - 1) / buf_duration if buf_duration > 0 else float(config.TARGET_FPS)
                print(
                    f"[Capture] Buffer has {len(clip0)} frames over {buf_duration:.1f}s"
                    f" = {actual_fps:.1f} actual fps  (configured {config.TARGET_FPS} fps)"
                )
            else:
                actual_fps = float(config.TARGET_FPS)

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

            report = self._run_analysis_pipeline(clip0, clip2, impact_frame_idx, actual_fps)
            self._results_queue.put(report)

        except Exception as e:
            import traceback
            print(f"[Analysis] Error: {e}")
            traceback.print_exc()
            self._results_queue.put(self._empty_report())

    def _run_analysis_pipeline(
        self,
        clip0: list,
        clip2: list,
        impact_frame_idx: int,
        actual_fps: float = 0.0,
    ) -> "SwingReport":
        """Shared pose + metrics pipeline used by both trigger paths."""
        self._analysis_progress = 0.10

        # Process cam0 (face-on) — all frames with temporal smoothing
        with PoseRunner() as runner0:
            results0 = runner0.process_clip(clip0)
        self._analysis_progress = 0.45

        # Process cam2 (down-the-line) — all frames with temporal smoothing
        with PoseRunner() as runner2:
            results2 = runner2.process_clip(clip2)
        self._analysis_progress = 0.65

        # Fuse dual-camera poses: for each landmark prefer the higher-confidence
        # camera, or weighted-average when both are moderate.  Fusion happens
        # AFTER temporal smoothing and BEFORE phase detection.
        results_fused = fuse_dual_camera_poses(results0, results2)
        self._analysis_progress = 0.70

        # Detect P1–P10 positions using fused results
        phases = detect_phases(results_fused, impact_frame_idx=impact_frame_idx)

        # Compute scorecard metrics and detailed position analysis from fused results
        metrics = compute_metrics(results_fused, results_fused, phases)
        swing_analysis = compute_swing_analysis(results_fused, results_fused, phases)
        self._analysis_progress = 0.85

        # Annotate frames with skeleton overlay using per-camera results (raw landmarks)
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
            actual_fps=actual_fps,
        )
        self._analysis_progress = 1.0
        print(f"[Analysis] Done — overall score: {metrics.overall_score}")
        print(f"[Analysis] Tempo: {metrics.tempo_ratio:.2f}:1  (BS {metrics.backswing_duration:.2f}s / DS {metrics.downswing_duration:.2f}s)")
        print(f"[Analysis] Address: {swing_analysis.address_evaluation}")
        print(f"[Analysis] Backswing: {swing_analysis.backswing_evaluation}")
        print(f"[Analysis] Transition: {swing_analysis.transition_evaluation}")
        print(f"[Analysis] Impact: {swing_analysis.impact_evaluation}")
        print(f"[Analysis] Follow-through: {swing_analysis.follow_through_evaluation}")
        return report

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
        self._manual_record_done.clear()
        self._manual_frames_cam0 = []
        self._manual_frames_cam2 = []
        self._manual_actual_fps = float(config.TARGET_FPS)

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
