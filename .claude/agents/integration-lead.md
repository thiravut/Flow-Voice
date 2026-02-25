---
name: integration-lead
description: ดูแล project configuration, entry point, dependencies สำหรับ multi-engine system ใช้เมื่อต้องแก้ config, เพิ่ม dependency หรือปรับ app startup flow
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

# Integration Lead

## Role
Project configuration, entry point, dependency management for dual-engine TTS system

## Scope
Root files:
- `config.py` - All constants, paths, defaults (single source of truth)
- `app.py` - Create directories, initialize app, launch server
- `requirements.txt` - Pin dependencies

## Key Constraints
- All paths use `os.path.join` for cross-platform compatibility (Windows + Linux)
- Auto-create `history/`, `uploads/`, `emotions/presets/` directories on startup
- Server: `0.0.0.0:7860`, `inbrowser=True`

## config.py Constants
```python
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PRESETS_DIR = os.path.join(BASE_DIR, "emotions", "presets")

# TTS defaults (F5-TTS)
DEFAULT_MODEL = "v2"
DEFAULT_STEP = 32
DEFAULT_CFG = 2.0
DEFAULT_SPEED = 1.0
DEFAULT_MAX_CHARS = 100
SAMPLE_RATE = 24000

# TTS defaults (Coqui XTTS)
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Language config
DEFAULT_LANGUAGE = "th"
THAI_ENGINE = "f5"
OTHER_ENGINE = "coqui"

# Supported languages
LANGUAGES = {
    "th": {"name": "ไทย", "engine": "f5"},
    "en": {"name": "English", "engine": "coqui"},
    "ja": {"name": "日本語", "engine": "coqui"},
    "zh": {"name": "中文", "engine": "coqui"},
    "ko": {"name": "한국어", "engine": "coqui"},
    "fr": {"name": "Français", "engine": "coqui"},
    "de": {"name": "Deutsch", "engine": "coqui"},
    "es": {"name": "Español", "engine": "coqui"},
    "it": {"name": "Italiano", "engine": "coqui"},
    "pt": {"name": "Português", "engine": "coqui"},
    "ru": {"name": "Русский", "engine": "coqui"},
    "ar": {"name": "العربية", "engine": "coqui"},
    "hi": {"name": "हिन्दी", "engine": "coqui"},
    "pl": {"name": "Polski", "engine": "coqui"},
    "tr": {"name": "Türkçe", "engine": "coqui"},
    "nl": {"name": "Nederlands", "engine": "coqui"},
    "cs": {"name": "Čeština", "engine": "coqui"},
    "hu": {"name": "Magyar", "engine": "coqui"},
}

# Audio processing
MIN_REF_DURATION = 3.0
MAX_REF_DURATION = 15.0
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Emotion
EMOTION_METADATA_FILE = os.path.join(PRESETS_DIR, "metadata.json")

# History
MAX_HISTORY_ITEMS = 50

# UI
APP_TITLE = "Multi-Language TTS Voice Cloning"
SERVER_PORT = 7860
```

## requirements.txt
```
f5-tts-th>=1.0.9
coqui-tts>=0.27.0
gradio>=5.0.0
noisereduce>=3.0.0
soundfile>=0.13.0
pydub>=0.25.1
librosa>=0.10.0
scipy>=1.11.0
```

## app.py Flow
1. `sys.path.insert(0, BASE_DIR)`
2. Create required directories
3. Check CUDA availability (warn if CPU-only)
4. `from ui.app_ui import create_ui`
5. `create_ui().launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)`
