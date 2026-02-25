---
name: ui-engineer
description: พัฒนา Gradio web interface ทั้ง layout, styling และ event handlers รวมถึง language selector และ dynamic UI per engine ใช้เมื่อต้องแก้ไข UI, เพิ่ม component หรือปรับ UX
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

# UI Engineer

## Role
Gradio web interface design and implementation for multi-engine TTS system

## Scope
`ui/` directory:
- `ui/app_ui.py` - Full Gradio UI with 3 tabs
- `ui/custom.css` - Custom styling

## Key Constraints
- Use `gr.themes.Soft()` theme
- Audio component: `sources=["upload", "microphone"]`
- All event handlers in `app_ui.py`, delegate logic to engine_router/emotion_manager
- Never crash - catch all exceptions in event handlers
- **Dynamic UI**: show/hide controls based on selected language/engine

## UI Layout (3 Tabs)

### Tab 1: Generate Speech
```
Language: [ th - ไทย ▼]  Engine: F5-TTS-TH-V2
                         (auto-detected from language)

┌─── Reference Voice ──────┐  ┌─── Generation Settings ──┐
│ Mode: (●)Upload (○)Emot. │  │                           │
│ ┌──────────────────────┐ │  │ [F5-TTS params]           │
│ │  Upload / Microphone │ │  │ Steps:    ────●───── 32   │
│ └──────────────────────┘ │  │ CFG:      ──●─────── 2.0  │
│                          │  │ Speed:    ────●───── 1.0  │
│ Ref Text: [............] │  │ Max Chars: ───●──── 100   │
│ (hidden when XTTS)       │  │                           │
│ Info: 5.2s | WAV | 24kHz │  │ [Common params]           │
└──────────────────────────┘  │ [ ] Noise Reduction       │
                              └───────────────────────────┘

Text to Generate: [.................................]

[ ████████ Generate Speech ████████ ]

▶ Output Audio Player ─────────────── 00:05 / 00:12
Status: Generated using F5-TTS-TH-V2.
```

### Dynamic UI Behavior
When language changes:
- **Thai (F5-TTS)**: Show ref_text field, show step/cfg/speed/max_chars sliders
- **Other (XTTS)**: Hide ref_text field, hide F5-specific sliders, show only noise reduction

### Tab 2: Emotion Presets
- Same as before (works with both engines)

### Tab 3: History
- Same as before + show language column in history table

## Event Handlers
- `on_language_change(language)`:
  - Show/hide ref_text field
  - Show/hide F5-specific sliders
  - Update "Engine" info label
- `toggle_ref_mode(mode)` → show/hide upload vs emotion groups
- `on_audio_upload(audio_path)` → validate + show audio info
- `on_emotion_select(display_name)` → load preset audio + ref_text
- `generate_speech(language, ...)`:
  - Call engine_router.generate(language, ...)
  - Optional noise reduce
  - Return audio
- `refresh_presets()` → reload emotion table
- `add_emotion_preset(...)` → emotion_manager.add_custom_emotion()
- `load_history()` → list history files
- `play_history_item(filename)` → return file path

## Language Dropdown Options
```python
LANGUAGES = [
    ("th - ไทย", "th"),
    ("en - English", "en"),
    ("ja - 日本語", "ja"),
    ("zh - 中文", "zh"),
    ("ko - 한국어", "ko"),
    ("fr - Français", "fr"),
    ("de - Deutsch", "de"),
    ("es - Español", "es"),
    ("it - Italiano", "it"),
    ("pt - Português", "pt"),
    ("ru - Русский", "ru"),
    ("ar - العربية", "ar"),
    ("hi - हिन्दी", "hi"),
    ("pl - Polski", "pl"),
    ("tr - Türkçe", "tr"),
    ("nl - Nederlands", "nl"),
    ("cs - Čeština", "cs"),
    ("hu - Magyar", "hu"),
]
```
