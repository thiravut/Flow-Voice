"""
config.py — Single source of truth for all constants and configuration.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent
HISTORY_DIR: Path = BASE_DIR / "history"
UPLOADS_DIR: Path = BASE_DIR / "uploads"
PRESETS_DIR: Path = BASE_DIR / "emotions" / "presets"

EMOTION_METADATA_FILE: Path = PRESETS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# TTS parameter defaults
# ---------------------------------------------------------------------------

INFERENCE_STEPS: int = 32
CFG_STRENGTH: float = 2.0
SPEED: float = 1.0
MAX_CHARS_CHUNK: int = 100
NOISE_REDUCTION: bool = False

# ---------------------------------------------------------------------------
# TTS parameter ranges (for UI sliders)
# ---------------------------------------------------------------------------

STEPS_MIN: int = 8
STEPS_MAX: int = 64

CFG_MIN: float = 0.5
CFG_MAX: float = 10.0

SPEED_MIN: float = 0.3
SPEED_MAX: float = 3.0

MAX_CHARS_MIN: int = 50
MAX_CHARS_MAX: int = 300

# ---------------------------------------------------------------------------
# Audio constraints
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS: list[str] = ["wav", "mp3", "ogg", "m4a", "flac"]
MIN_AUDIO_DURATION: float = 3.0    # seconds
MAX_AUDIO_DURATION: float = 15.0   # seconds
OUTPUT_SAMPLE_RATE: int = 24000    # Hz

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "th": "ไทย",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "हिन्दी",
    "pl": "Polski",
    "tr": "Türkçe",
    "nl": "Nederlands",
    "cs": "Čeština",
    "hu": "Magyar",
}

DEFAULT_LANGUAGE: str = "th"

LANGUAGE_ENGINE_MAP: dict[str, str] = {
    lang: ("f5" if lang == "th" else "coqui")
    for lang in SUPPORTED_LANGUAGES
}

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

ENGINE_NAMES: dict[str, str] = {
    "f5": "F5-TTS-TH-V2",
    "coqui": "Coqui XTTS v2",
}

F5_MODEL_VERSION: str = "v2"
XTTS_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

MAX_HISTORY: int = 50

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 7860

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

APP_TITLE: str = "Multi-Language TTS Voice Cloning"
