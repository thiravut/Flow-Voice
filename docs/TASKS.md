# Task List — Multi-Language TTS Voice Cloning Web UI

> Auto-generated from `docs/PRD.md`
> Date: 2026-02-25

---

## Phase 1: Foundation

### Task 1.1: Create Directory Structure
- **Size**: S
- **Agent**: integration-lead
- **Files**:
  - `engine/__init__.py` (create)
  - `ui/__init__.py` (create)
  - `emotions/__init__.py` (create)
  - `emotions/presets/` (create dir)
  - `history/` (create dir)
  - `uploads/` (create dir)
- **Depends on**: None
- **Acceptance Criteria**: Directory structure matches PRD Section 4.1 exactly. All `__init__.py` files exist. `history/` and `uploads/` directories are auto-creatable at runtime.
- **Details**: Create the full directory tree as specified in PRD 4.1. Add `.gitkeep` in `history/` and `uploads/` so Git tracks empty dirs. The `__init__.py` files can be empty for now; they will be populated when each module is implemented.

---

### Task 1.2: Create `requirements.txt`
- **Size**: S
- **Agent**: integration-lead
- **Files**:
  - `requirements.txt` (create)
- **Depends on**: None
- **Acceptance Criteria**: All dependencies listed in PRD Section 2 are included with pinned or minimum versions: `f5-tts-th`, `coqui-tts` (TTS), `gradio>=5.0`, `soundfile`, `pydub`, `librosa`, `noisereduce`, `torch>=2.2`.
- **Details**: Group dependencies by purpose (TTS engines, UI, audio processing, runtime). Add comments explaining each group. Ensure Python >= 3.10 compatibility. Note: FFmpeg is a system-level dependency and should be documented separately, not in requirements.txt.

---

### Task 1.3: Create `config.py`
- **Size**: M
- **Agent**: integration-lead
- **Files**:
  - `config.py` (create)
- **Depends on**: Task 1.1
- **Acceptance Criteria**:
  - All parameter defaults from PRD 3.3 defined: `INFERENCE_STEPS=32`, `CFG_STRENGTH=2.0`, `SPEED=1.0`, `MAX_CHARS_CHUNK=100`, `NOISE_REDUCTION=False`
  - Parameter ranges defined: Steps (8-64), CFG (0.5-10.0), Speed (0.3-3.0), Max Chars (50-300)
  - Supported languages mapping (language code -> engine name) from PRD 3.2
  - Audio constraints: supported formats (`WAV, MP3, OGG, M4A, FLAC`), min/max duration (3-15s), output sample rate (24kHz)
  - File paths: `HISTORY_DIR`, `UPLOADS_DIR`, `PRESETS_DIR`
  - History limit: `MAX_HISTORY=50`
  - Output format: WAV 24kHz
  - `LANGUAGE_ENGINE_MAP` dict mapping all 18 languages to their engine
- **Details**: This is the **single source of truth** for all constants. Every other module must import from here. Include a `LANGUAGES` dict with display names (e.g., `{"th": "ไทย", "en": "English", "ja": "日本語", ...}`). Include an `ENGINE_NAMES` dict (e.g., `{"f5": "F5-TTS-TH-V2", "coqui": "Coqui XTTS v2"}`). All paths should use `pathlib.Path` for cross-platform compatibility.

---

## Phase 2: Core Engine Layer

### Task 2.1: Create `engine/base_engine.py` — Abstract Base Class
- **Size**: M
- **Agent**: engine-architect
- **Files**:
  - `engine/base_engine.py` (create)
- **Depends on**: Task 1.3
- **Acceptance Criteria**:
  - Abstract class `BaseTTSEngine` with `abc.ABC`
  - Abstract methods: `load_model()`, `unload_model()`, `generate(ref_audio, gen_text, **kwargs) -> str` (returns output path)
  - Property: `is_loaded -> bool`
  - Thread lock mechanism (PRD 4.4: "Thread lock — Gradio uses threading, prevent CUDA OOM")
  - Common error handling pattern
- **Details**: Define the interface contract that both F5Engine and CoquiEngine must implement. The `generate()` method should accept `ref_audio` (path to reference audio file), `gen_text` (text to generate), and engine-specific kwargs. Include a `threading.Lock` as a class attribute so that only one engine generates at a time (preventing CUDA OOM). Add a context manager or decorator for thread-safe generation. The base class should also handle saving output to `history/` with the naming convention from PRD 3.5 (`YYYYMMDD_HHMMSS_<hash>.wav`).

---

### Task 2.2: Create `engine/audio_processor.py` — Shared Audio Utilities
- **Size**: M
- **Agent**: engine-architect
- **Files**:
  - `engine/audio_processor.py` (create)
- **Depends on**: Task 1.3
- **Acceptance Criteria**:
  - Validate audio file format (WAV, MP3, OGG, M4A, FLAC) — PRD 3.1
  - Validate audio duration (3-15 seconds) with clear error messages — PRD 3.1
  - Return audio info (duration, format, sample rate) — PRD 3.1
  - Noise reduction function (optional, using `noisereduce`) — PRD 3.3
  - Text chunking by `max_chars_chunk` parameter — PRD 3.3
  - Audio concatenation for chunked generation
- **Details**: Use `librosa` for audio loading and analysis, `soundfile` for WAV I/O, `pydub` for format conversion, `noisereduce` for optional noise reduction. The `validate_audio()` function should return a dict: `{"valid": bool, "duration": float, "format": str, "sample_rate": int, "error": str|None}`. The `chunk_text()` function should split text respecting sentence/word boundaries. Implement `apply_noise_reduction(audio_path) -> audio_path`. Provide `get_audio_info(path) -> dict` for UI display.

---

### Task 2.3: Create `engine/f5_engine.py` — F5-TTS-TH Wrapper
- **Size**: L
- **Agent**: ai-engineer
- **Files**:
  - `engine/f5_engine.py` (create)
- **Depends on**: Task 2.1, Task 2.2
- **Acceptance Criteria**:
  - Extends `BaseTTSEngine`
  - Loads F5-TTS-TH-V2 model (auto-download from HuggingFace on first run)
  - `generate()` accepts: `ref_audio`, `ref_text` (optional), `gen_text`, `steps`, `cfg_strength`, `speed`, `max_chars_chunk`
  - Supports `ref_text` as optional field — PRD 3.1
  - Output: WAV 24kHz — PRD 2
  - Model unload frees VRAM — PRD 4.4
  - Text chunking for long text (via `audio_processor`) — PRD 3.3
  - Thread-safe generation (via base class lock) — PRD 4.4
- **Details**: Wrap the `f5-tts-th` package. The F5 engine is phoneme-based and uses diffusion transformer + flow matching. It needs `ref_audio` (3-15s) and optionally `ref_text`. Implement `load_model()` to load onto GPU, `unload_model()` to move to CPU or delete and free CUDA cache. For long text, chunk using `audio_processor.chunk_text()`, generate each chunk, then concatenate. Apply speed adjustment post-generation if supported by the library, or via parameter. Map PRD parameters: `steps` -> inference steps, `cfg_strength` -> CFG guidance.

---

### Task 2.4: Create `engine/coqui_engine.py` — Coqui XTTS v2 Wrapper
- **Size**: L
- **Agent**: ai-engineer
- **Files**:
  - `engine/coqui_engine.py` (create)
- **Depends on**: Task 2.1, Task 2.2
- **Acceptance Criteria**:
  - Extends `BaseTTSEngine`
  - Loads Coqui XTTS v2 model (auto-download on first run)
  - `generate()` accepts: `ref_audio`, `gen_text`, `language`
  - Does NOT require `ref_text` — PRD 2.1, 3.2
  - Supports 17 languages — PRD 3.2
  - Cross-lingual voice cloning — PRD 3.2
  - Output: WAV 24kHz — PRD 2
  - Model unload frees VRAM — PRD 4.4
  - Thread-safe generation (via base class lock) — PRD 4.4
- **Details**: Wrap the `TTS` package from Coqui. XTTS v2 uses GPT + VQ-VAE architecture. It only needs `ref_audio` (3-6s, but we accept up to 15s) and does NOT need `ref_text`. The `language` parameter maps to XTTS language codes. Cross-lingual cloning works automatically (ref audio in one language, generate in another). Note: XTTS does NOT use `steps` or `cfg_strength` parameters, so those sliders should be hidden in UI when this engine is active (handled in UI layer). Speed parameter may be handled differently than F5.

---

### Task 2.5: Create `engine/engine_router.py` — Language-Based Router
- **Size**: M
- **Agent**: engine-architect
- **Files**:
  - `engine/engine_router.py` (create)
- **Depends on**: Task 2.3, Task 2.4
- **Acceptance Criteria**:
  - Routes to F5Engine when `language == "th"` — PRD 4.3
  - Routes to CoquiEngine for all other 17 languages — PRD 4.3
  - VRAM swap: unloads inactive engine when loading new one — PRD 4.3, 4.4
  - Lazy loading: engines loaded on first use, not at startup
  - Single entry point: `router.generate(language, ref_audio, gen_text, **kwargs) -> str`
  - Engine swap latency handled gracefully (status feedback) — PRD 8
- **Details**: Implement `EngineRouter` class that holds references to both engines. Use the `LANGUAGE_ENGINE_MAP` from `config.py` to determine routing. Implement VRAM swap strategy: when switching from F5 to Coqui (or vice versa), unload the old engine first, then load the new one. Track `current_engine` to avoid unnecessary reloads. Provide a `get_current_engine_name(language) -> str` method for UI to display. The router should expose which parameters are relevant per engine so UI can adapt sliders.

---

### Task 2.6: Update `engine/__init__.py` — Public API
- **Size**: S
- **Agent**: engine-architect
- **Files**:
  - `engine/__init__.py` (modify)
- **Depends on**: Task 2.5
- **Acceptance Criteria**:
  - Exports `EngineRouter`, `AudioProcessor`
  - Clean public API for UI layer
- **Details**: Import and re-export the key classes so the UI layer can do `from engine import EngineRouter, AudioProcessor`. Keep internal classes (BaseTTSEngine, F5Engine, CoquiEngine) as implementation details.

---

## Phase 3: Emotion System

### Task 3.1: Create `emotions/presets/metadata.json` — Default Presets
- **Size**: S
- **Agent**: emotion-designer
- **Files**:
  - `emotions/presets/metadata.json` (create)
- **Depends on**: Task 1.1
- **Acceptance Criteria**:
  - 6 default emotion entries: neutral, happy, sad, angry, excited, calm — PRD 3.4
  - Each entry has: `id`, `name_en`, `name_th`, `description`, `ref_text`, `audio_file` (null by default)
  - Note in PRD: "ระบบจัดส่งพร้อม metadata ของ 6 อารมณ์ แต่ไม่มีไฟล์เสียง ผู้ใช้ต้องบันทึกหรืออัพโหลดเสียง reference เอง" — audio files are null initially
- **Details**: Create JSON structure like:
  ```json
  {
    "presets": [
      {
        "id": "neutral",
        "name_en": "Neutral",
        "name_th": "ปกติ",
        "description": "น้ำเสียงปกติ สมดุล",
        "ref_text": "",
        "audio_file": null,
        "is_default": true
      }
    ]
  }
  ```
  Use the names and descriptions from PRD 3.4 table. The `audio_file` field will be a relative path (or null). Default presets are marked `is_default: true` and cannot be deleted.

---

### Task 3.2: Create `emotions/emotion_manager.py` — Preset CRUD
- **Size**: M
- **Agent**: emotion-designer
- **Files**:
  - `emotions/emotion_manager.py` (create)
- **Depends on**: Task 3.1, Task 1.3
- **Acceptance Criteria**:
  - List all presets (default + custom) — PRD 3.4
  - Get preset by ID — PRD 3.4
  - Add custom preset (name + audio + ref_text) — PRD 3.4
  - Delete custom preset (not default presets) — PRD 3.4
  - Preview: return audio path for a preset — PRD 3.4
  - Return preset data as table-ready format for UI — PRD 3.4
- **Details**: Implement `EmotionManager` class that loads `metadata.json` on init. Methods: `list_presets() -> list[dict]`, `get_preset(id) -> dict`, `add_preset(name, audio_path, ref_text) -> dict`, `delete_preset(id) -> bool`, `get_presets_table() -> list[list]` (for Gradio DataFrame). Custom presets store audio files in `emotions/presets/`. File naming: `custom_<name>_<hash>.wav`. Persist changes back to `metadata.json`. Validate that audio exists before returning preset.

---

### Task 3.3: Update `emotions/__init__.py` — Public API
- **Size**: S
- **Agent**: emotion-designer
- **Files**:
  - `emotions/__init__.py` (modify)
- **Depends on**: Task 3.2
- **Acceptance Criteria**:
  - Exports `EmotionManager`
- **Details**: Import and re-export `EmotionManager` for clean API access from UI layer.

---

## Phase 4: Web UI

### Task 4.1: Create `ui/custom.css` — Styling
- **Size**: S
- **Agent**: ui-engineer
- **Files**:
  - `ui/custom.css` (create)
- **Depends on**: None
- **Acceptance Criteria**:
  - Clean, readable UI — PRD 5
  - Tab layout styling
  - Status/info text styling
  - Responsive layout (reference voice + generation settings side by side) — PRD 5 mockup
- **Details**: Based on the PRD 5 mockup, the UI has a two-column layout in the Generate tab: left column for "Reference Voice" and right column for "Generation Settings". Use Gradio CSS variables for consistent theming. Keep styling minimal since Gradio handles most layout.

---

### Task 4.2: Create `ui/app_ui.py` — Tab 1: Generate Speech
- **Size**: L
- **Agent**: ui-engineer
- **Files**:
  - `ui/app_ui.py` (create)
- **Depends on**: Task 2.5, Task 2.6, Task 1.3
- **Acceptance Criteria**:
  - Language dropdown with all 18 languages (default: Thai) — PRD 3.2
  - Display active engine name based on selected language — PRD 3.2
  - Reference audio upload (drag & drop + microphone recording) — PRD 3.1
  - Display audio info (duration, format, sample rate) — PRD 3.1
  - Audio duration validation (3-15s) with warning — PRD 3.1
  - Supported formats: WAV, MP3, OGG, M4A, FLAC — PRD 3.1
  - `ref_text` field: visible for F5-TTS (Thai), hidden for XTTS — PRD 3.2
  - Toggle between "Upload Audio" and "Emotion Preset" mode — PRD 3.4
  - Parameter sliders: Steps, CFG, Speed, Max Chars, Noise Reduction — PRD 3.3
  - Sliders adapt per engine (Steps/CFG shown only for F5) — PRD 3.2
  - Recommended values display — PRD 3.3
  - Text input area — PRD 5
  - Generate button — PRD 5
  - Output audio player — PRD 3.1
  - Status message — PRD 5
- **Details**: This is the main generation tab. Use `gr.Blocks` with `gr.Tab`. The language dropdown triggers: (1) update engine display label, (2) show/hide `ref_text`, (3) show/hide F5-specific sliders. The "mode toggle" switches between audio upload component and emotion preset dropdown. Generate button calls `engine_router.generate()`. Show loading indicator during generation. Validate inputs before generation: check text is not empty, check ref audio is provided and valid.

---

### Task 4.3: Add to `ui/app_ui.py` — Tab 2: Emotion Presets
- **Size**: M
- **Agent**: ui-engineer
- **Files**:
  - `ui/app_ui.py` (modify)
- **Depends on**: Task 4.2, Task 3.2
- **Acceptance Criteria**:
  - Dropdown to select emotion preset — PRD 3.4
  - Preview audio button for selected preset — PRD 3.4
  - Display `ref_text` of selected preset — PRD 3.4
  - Table overview of all presets — PRD 3.4
  - Add custom preset form: name + audio upload + ref_text — PRD 3.4
  - Delete custom preset button — PRD 3.4
- **Details**: Second tab in the UI. Top section: preset table (Gradio DataFrame) showing all presets with columns (Name, Name TH, Description, Has Audio, ref_text). Bottom section: two-column layout — left for "Add Custom Preset" form (text input for name, audio upload, text input for ref_text, Add button), right for "Delete Preset" (dropdown of custom presets + Delete button). Preview uses `gr.Audio` component. Connect to `EmotionManager` methods.

---

### Task 4.4: Add to `ui/app_ui.py` — Tab 3: History
- **Size**: M
- **Agent**: ui-engineer
- **Files**:
  - `ui/app_ui.py` (modify)
- **Depends on**: Task 4.2
- **Acceptance Criteria**:
  - List generated audio files from `history/` — PRD 3.5
  - Display: filename, date, file size — PRD 3.5
  - Play audio from history — PRD 3.5
  - Maximum 50 entries — PRD 3.5
  - Filename format: `YYYYMMDD_HHMMSS_<hash>.wav` — PRD 3.5
- **Details**: Third tab. Use Gradio DataFrame to show history list, sorted by date (newest first). On row select, load audio into `gr.Audio` player. Implement `load_history()` function that scans `history/` directory, parses filenames for timestamp, gets file size, and returns as list. Cap at 50 entries (defined in `config.py`). Add a "Refresh" button to reload the list.

---

### Task 4.5: Update `ui/__init__.py` — Public API
- **Size**: S
- **Agent**: ui-engineer
- **Files**:
  - `ui/__init__.py` (modify)
- **Depends on**: Task 4.4
- **Acceptance Criteria**:
  - Exports the main UI builder function
- **Details**: Export the `create_ui()` function (or equivalent) that `app.py` will call to build and launch the Gradio app.

---

## Phase 5: Integration & Testing

### Task 5.1: Create `app.py` — Entry Point
- **Size**: S
- **Agent**: integration-lead
- **Files**:
  - `app.py` (create)
- **Depends on**: Task 4.5
- **Acceptance Criteria**:
  - `python app.py` launches browser at `http://localhost:7860` — PRD 7
  - Initializes EngineRouter, EmotionManager
  - Passes dependencies to UI builder
  - Auto-creates `history/` and `uploads/` dirs if missing — PRD 4.1
- **Details**: Minimal entry point. Import `create_ui` from `ui`, instantiate `EngineRouter` and `EmotionManager`, pass them to the UI builder, then call `demo.launch(server_port=7860)`. Add argument parsing for optional port/share settings. Ensure directories exist on startup.

---

### Task 5.2: End-to-End Test — Thai (F5-TTS)
- **Size**: M
- **Agent**: qa-tester
- **Files**:
  - (no new files — manual testing)
- **Depends on**: Task 5.1
- **Acceptance Criteria**:
  - Select Thai + Upload WAV + Generate -> Thai speech from F5-TTS — PRD 7
  - Steps slider visible and functional — PRD 7
  - CFG slider visible and functional — PRD 7
  - ref_text field visible — PRD 3.2
  - Output saved to `history/` — PRD 3.5
  - Upload file < 3s -> warning "Audio too short" — PRD 7
  - Empty text + Generate -> warning "Please enter text" — PRD 7
- **Details**: Manually test the full Thai generation flow. Verify F5-TTS model downloads on first run. Check that all parameter sliders affect output. Confirm history saving with correct filename format. Test edge cases: too-short audio, no text, very long text (chunking).

---

### Task 5.3: End-to-End Test — Multi-Language (Coqui XTTS)
- **Size**: M
- **Agent**: qa-tester
- **Files**:
  - (no new files — manual testing)
- **Depends on**: Task 5.1
- **Acceptance Criteria**:
  - Select English + Upload WAV + Generate -> English speech from XTTS — PRD 7
  - Select Japanese + Generate -> Japanese speech from XTTS — PRD 7
  - ref_text field hidden when non-Thai language selected — PRD 7
  - Steps/CFG sliders hidden for XTTS languages — PRD 3.2
  - Cross-lingual: ref=Thai audio, gen=English -> cloned voice speaks English — PRD 7
- **Details**: Test multiple languages (at minimum: English, Japanese, Chinese). Verify XTTS model downloads on first run. Test cross-lingual cloning specifically. Confirm that engine-specific parameters are correctly hidden/shown.

---

### Task 5.4: End-to-End Test — Engine Swap & VRAM
- **Size**: M
- **Agent**: qa-tester
- **Files**:
  - (no new files — manual testing)
- **Depends on**: Task 5.2, Task 5.3
- **Acceptance Criteria**:
  - Switch Thai -> English -> Thai without crash — PRD 7
  - VRAM is freed when swapping engines — PRD 4.3, 4.4
  - Engine swap latency is handled (UI shows loading state) — PRD 8
- **Details**: Test rapid engine switching. Monitor VRAM usage (nvidia-smi) to verify unloading works. Test generating immediately after swap. Ensure no CUDA OOM errors.

---

### Task 5.5: End-to-End Test — Emotion System
- **Size**: M
- **Agent**: qa-tester
- **Files**:
  - (no new files — manual testing)
- **Depends on**: Task 5.1
- **Acceptance Criteria**:
  - Add custom emotion preset (upload audio + ref_text) -> appears in list — PRD 7
  - Generate with emotion preset -> output reflects emotion — PRD 7
  - Delete custom preset -> removed from list — PRD 3.4
  - Default presets cannot be deleted — PRD 3.4
  - Emotion presets table displays correctly — PRD 3.4
- **Details**: Test the full emotion workflow. Add a custom emotion, generate speech using it, then delete it. Verify that the 6 default presets are listed but flagged as "no audio" initially. Test preview functionality.

---

### Task 5.6: End-to-End Test — History
- **Size**: S
- **Agent**: qa-tester
- **Files**:
  - (no new files — manual testing)
- **Depends on**: Task 5.2
- **Acceptance Criteria**:
  - Multiple generations -> files appear in History tab — PRD 7
  - History shows filename, date, size — PRD 3.5
  - Audio playback from history works — PRD 3.5
  - Maximum 50 entries enforced — PRD 3.5
- **Details**: Generate several audio files, then check History tab. Verify sorting (newest first), file info accuracy, and playback. Test the 50-entry limit behavior.

---

## Dependency Graph

```
Task 1.1 ─────┬────────────────────────────────────────────► Task 3.1
              │
Task 1.2      │
              │
Task 1.3 ◄────┘
  │
  ├──► Task 2.1 (base_engine) ──┬──► Task 2.3 (f5_engine) ──┐
  │                             │                            │
  ├──► Task 2.2 (audio_proc) ──┤                            ├──► Task 2.5 (router) ──► Task 2.6
  │                             │                            │
  │                             └──► Task 2.4 (coqui_engine)─┘
  │
  └──► Task 4.1 (css)

Task 3.1 ──► Task 3.2 ──► Task 3.3

Task 2.6 ──┬──► Task 4.2 (Tab 1) ──┬──► Task 4.3 (Tab 2) ──► Task 4.4 (Tab 3) ──► Task 4.5
           │                       │
Task 3.3 ──┘                       │
                                   │
Task 4.5 ──► Task 5.1 (app.py) ──┬──► Task 5.2 (test Thai)
                                  ├──► Task 5.3 (test multi-lang)
                                  ├──► Task 5.5 (test emotion)
                                  └──► Task 5.6 (test history)

Task 5.2 + Task 5.3 ──► Task 5.4 (test swap)
```

---

## Summary

| Phase | Tasks | Size Breakdown | Key Agent |
|-------|-------|---------------|-----------|
| Phase 1: Foundation | 3 tasks | 2S + 1M | integration-lead |
| Phase 2: Core Engine | 6 tasks | 2S + 2M + 2L | engine-architect, ai-engineer |
| Phase 3: Emotion System | 3 tasks | 2S + 1M | emotion-designer |
| Phase 4: Web UI | 5 tasks | 2S + 2M + 1L | ui-engineer |
| Phase 5: Integration | 6 tasks | 2S + 4M | integration-lead, qa-tester |
| **Total** | **23 tasks** | **8S + 10M + 3L** | |

---

## Clarification Needed

| Item | Question | PRD Reference |
|------|----------|---------------|
| F5-TTS ref_text | PRD says ref_text is "optional" for F5-TTS. What happens if ref_text is not provided? Does F5-TTS auto-transcribe, or does quality degrade? This affects UI guidance. | PRD 3.1 |
| Speed parameter for XTTS | PRD lists Speed (0.3-3.0) as a general parameter, but XTTS v2 may handle speed differently than F5. Should Speed slider be shown for XTTS, or only for F5? | PRD 3.3 |
| Noise reduction scope | Is noise reduction applied to the final output only, or also to the reference audio before cloning? | PRD 3.3 |
| History auto-cleanup | PRD says max 50 entries. When the 51st is created, should the oldest be auto-deleted, or should generation be blocked? | PRD 3.5 |
| Emotion presets language scope | Are emotion presets only for Thai (F5-TTS), or should they also work with XTTS languages? The PRD mentions "F5-TTS clones voice and emotion from ref audio" but XTTS does similar. | PRD 3.4 |
