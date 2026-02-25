# Multi-Language TTS Voice Cloning - Project Instructions

## Project Overview
ระบบสร้างเสียงพูดหลายภาษาผ่าน Web UI ใช้ 2 TTS engines:
- **F5-TTS-TH-V2** — ภาษาไทย (คุณภาพสูงสุด)
- **Coqui XTTS v2** — 17 ภาษาอื่น (EN, JA, ZH, KO, FR, DE, ...)

ระบบ route อัตโนมัติตามภาษาที่เลือก

- PRD: `docs/PRD.md`
- Entry point: `app.py`
- UI: Gradio (port 7860)

## Tech Stack
- Python >= 3.10, PyTorch >= 2.2 (CUDA)
- `f5-tts-th` (Thai TTS), `coqui-tts` (Multi-lang TTS)
- Gradio >= 5.0
- Audio: soundfile, pydub, librosa, noisereduce

## Project Structure
```
voice/
├── app.py                   # Entry point
├── config.py                # Constants, languages, defaults
├── requirements.txt
├── engine/
│   ├── base_engine.py       # Abstract base class (ABC)
│   ├── f5_engine.py         # F5-TTS-TH-V2 (Thai)
│   ├── coqui_engine.py      # Coqui XTTS v2 (17 langs)
│   ├── engine_router.py     # Language → engine routing + VRAM swap
│   └── audio_processor.py   # Shared audio utilities
├── ui/
│   ├── app_ui.py            # Gradio UI (3 tabs) + lang selector
│   └── custom.css
├── emotions/
│   ├── emotion_manager.py   # Emotion preset CRUD
│   └── presets/metadata.json
├── history/                 # Generated audio (auto-created)
└── uploads/                 # User uploads (auto-created)
```

## Coding Conventions

### Python Style
- Python 3.10+ (type hints with `|`, match statements)
- Type hints for function signatures
- Docstrings for public methods only (single line preferred)
- `os.path.join` for all file paths (Windows compatibility)

### Error Handling
- Validate at boundaries (user input, file uploads)
- User-friendly error messages
- Never crash Gradio server - catch exceptions in event handlers

### File Naming
- Snake_case for Python files
- Generated audio: `{YYYYMMDD}_{HHMMSS}_{hash8}.wav`
- Custom emotion presets: `custom_{name}.wav`

## Architecture Key Rules
- **VRAM swap**: Only ONE engine loaded at a time. Unload before loading another.
- **Singleton router**: Single EngineRouter instance with thread lock
- **Lazy loading**: Don't load models until first generate request
- **Language routing**: `th` → F5Engine, everything else → CoquiEngine
- **Shared output**: Both engines output WAV 24kHz

## Common Commands
```bash
pip install -r requirements.txt
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
python app.py
```
