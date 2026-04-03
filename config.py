# ── Camera ────────────────────────────────────────────────────────────────────
CAM0_INDEX = 0          # Face-on camera
CAM2_INDEX = 2          # Down-the-line camera
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
TARGET_FPS = 60
BUFFER_SECONDS = 6.0
POST_TRIGGER_SECONDS = 3.0

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_BLOCKSIZE = 1024
TRIGGER_RMS_THRESHOLD = 0.15   # 0.0–1.0 float32 amplitude; tune per environment
TRIGGER_COOLDOWN_S = 2.0
AUDIO_DEVICE_INDEX = None      # None = system default; set to int if needed

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_MODEL_COMPLEXITY = 0        # 0=Lite, 1=Full, 2=Heavy
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
RENDER_FPS = 60

# Layout pixel constants
HEADER_HEIGHT = 60
METRICS_HEIGHT = 150
CONTROLS_HEIGHT = 60
VIDEO_AREA_HEIGHT = DISPLAY_HEIGHT - HEADER_HEIGHT - METRICS_HEIGHT - CONTROLS_HEIGHT  # 810
VIDEO_CELL_WIDTH = DISPLAY_WIDTH // 2  # 960

# Video frame display size within each cell (letterboxed)
VIDEO_DISPLAY_WIDTH = 720
VIDEO_DISPLAY_HEIGHT = 540

# ── Review-mode layout (VIDEO_AREA_HEIGHT is split into these three sections) ──
TIMELINE_HEIGHT     = 70   # P1–P10 scrub bar
P_DETAIL_HEIGHT     = 90   # per-position body-angle detail strip
REVIEW_VIDEO_HEIGHT = (
    DISPLAY_HEIGHT - HEADER_HEIGHT
    - TIMELINE_HEIGHT - P_DETAIL_HEIGHT
    - METRICS_HEIGHT - CONTROLS_HEIGHT
)  # 1080 - 60 - 70 - 90 - 150 - 60 = 650
# Sanity: REVIEW_VIDEO_HEIGHT + TIMELINE_HEIGHT + P_DETAIL_HEIGHT == VIDEO_AREA_HEIGHT (810)

# ── Colors (dark sports-display theme) ───────────────────────────────────────
COLOR_BG           = (26,  26,  46)    # #1a1a2e dark navy
COLOR_PANEL        = (22,  33,  62)    # #16213e header/panel bg
COLOR_ACCENT       = (108, 92, 231)    # #6c5ce7 purple accent
COLOR_TEXT         = (220, 220, 220)   # near-white
COLOR_TEXT_DIM     = (130, 130, 150)   # dimmed label
COLOR_GOOD         = (0,  184, 148)    # #00b894 green  (score ≥ 80)
COLOR_WARN         = (253, 203, 110)   # #fdcb6e amber  (60–79)
COLOR_BAD          = (214,  48,  49)   # #d63031 red    (< 60)
COLOR_POSE_SKELETON= (0,  206, 201)    # #00cec9 cyan overlay
COLOR_RECORD_DOT   = (214,  48,  49)   # pulsing red
COLOR_BUTTON_BG    = (40,  40,  70)
COLOR_BUTTON_HOVER = (60,  60, 100)
COLOR_BORDER       = (50,  50,  80)

# ── Scoring thresholds ────────────────────────────────────────────────────────
SCORE_GREEN_THRESHOLD = 80
SCORE_AMBER_THRESHOLD = 60

# ── Spine angle (degrees forward tilt from vertical, face-on view) ────────────
SPINE_ANGLE_IDEAL = 35.0
SPINE_ANGLE_TOLERANCE = 10.0   # ±10° from ideal = full score

# ── Hip rotation (degrees, face-on) ──────────────────────────────────────────
HIP_ROTATION_IDEAL = 45.0      # ≥45° = full score

# ── Knee flex (joint angle at address, face-on) ──────────────────────────────
KNEE_FLEX_IDEAL_MIN = 150.0
KNEE_FLEX_IDEAL_MAX = 165.0

# ── Head stability (std-dev of nose landmark across address→impact) ───────────
HEAD_STABILITY_MAX_STDDEV = 0.04   # normalised coords; lower = better

# ── Analysis ──────────────────────────────────────────────────────────────────
# Drop to 15 FPS for pose analysis to speed up processing
ANALYSIS_SAMPLE_FPS = 15

# ── Swing tempo ───────────────────────────────────────────────────────────────
# Tempo = backswing duration / downswing duration  (tour average ≈ 3.0)
TEMPO_TARGET = 3.0
TEMPO_GREEN_MIN = 2.7    # green band (2.7–3.3)
TEMPO_GREEN_MAX = 3.3
TEMPO_AMBER_MIN = 2.5    # amber band: outside green but within ~17%
TEMPO_AMBER_MAX = 3.5

# ── Manual trigger (spacebar) ─────────────────────────────────────────────────
MANUAL_COUNTDOWN_SECONDS = 3   # countdown before recording starts
MANUAL_RECORD_SECONDS    = 6   # how long to record after countdown

# ── Save directory ────────────────────────────────────────────────────────────
SAVE_DIR = "/home/brian/GolfSwings"
