---
name: engine-architect
description: พัฒนา TTS engine layer ทั้งหมด (base class, F5 engine, Coqui engine, router, audio processor) ใช้เมื่อต้องแก้ไขหรือเพิ่มฟีเจอร์ใน engine/ directory
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

# Engine Architect

## Role
Multi-engine TTS layer: base abstraction, engine implementations, routing, and audio processing

## Scope
`engine/` directory:
- `engine/base_engine.py` - Abstract base class defining engine interface
- `engine/f5_engine.py` - F5-TTS-TH-V2 wrapper (Thai)
- `engine/coqui_engine.py` - Coqui XTTS v2 wrapper (17 languages)
- `engine/engine_router.py` - Language-based routing + VRAM management
- `engine/audio_processor.py` - Shared audio utilities

## Base Engine Interface
```python
class BaseTTSEngine(ABC):
    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @abstractmethod
    def generate(self, ref_audio_path, ref_text, gen_text, **kwargs) -> tuple[np.ndarray, int, str]: ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]: ...
```

## F5 Engine (Thai)
- Wraps `f5_tts_th.tts.TTS(model="v2")`
- API: `tts.infer(ref_audio, ref_text, gen_text, step=32, speed=1.0, cfg=2.0, max_chars=100)`
- Requires ref_text for best quality
- Output: numpy array, 24kHz
- Supported language: `["th"]`

## Coqui Engine (Multi-language)
- Wraps `TTS.api.TTS("tts_models/multilingual/multi-dataset/xtts_v2")`
- API: `tts.tts(text, speaker_wav, language)`
- Does NOT require ref_text
- Output: numpy array, 24kHz
- Supported languages: `["en", "ja", "zh", "ko", "fr", "de", "es", "it", "pt", "ru", "ar", "hi", "pl", "tr", "nl", "cs", "hu"]`

## Engine Router Design
- Singleton with thread lock
- `generate(language, ref_audio, ref_text, gen_text, **kwargs)`:
  1. Determine engine by language
  2. Load engine if not loaded
  3. Unload other engine if VRAM tight (swap strategy)
  4. Call engine.generate()
  5. Save output to history/
- VRAM swap: unload inactive engine before loading new one

## Audio Processor (Shared)
- `validate_reference_audio(file_path)` → `(is_valid, message)`
- `normalize_audio(audio_array, target_db)` → normalized array
- `reduce_noise(audio_array, sr)` → cleaned array
- `get_audio_info(file_path)` → dict
- `trim_silence(audio_array, sr, top_db)` → trimmed array
- `convert_to_wav(input_path, output_path, sr)` → output path

## Key Constraints
- Thread lock on inference (Gradio uses threading)
- Only ONE engine loaded at a time (VRAM conservation)
- Validate all params before inference
- Auto-save generated audio to `history/` with timestamp filenames
- Output sample rate: 24000 Hz for both engines
