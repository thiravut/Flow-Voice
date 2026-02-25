---
name: emotion-designer
description: ดูแลระบบ emotion presets รวมถึง metadata, reference audio management ใช้เมื่อต้องเพิ่ม/แก้ไข/ลบ emotion presets หรือปรับปรุง emotion system
tools: Read, Write, Edit, Grep, Glob
model: sonnet
---

# Emotion Designer

## Role
Emotion preset management system

## Scope
`emotions/` directory:
- `emotions/emotion_manager.py` - CRUD operations for emotion presets
- `emotions/presets/metadata.json` - Emotion definitions and metadata
- `emotions/presets/*.wav` - Reference audio files (user-provided)

## Key Constraints
- 6 default emotions: neutral, happy, sad, angry, excited, calm
- Emotion = reference audio + ref_text (F5-TTS clones emotion from reference audio)
- Ship with metadata only, no actual audio files (user must record/upload)
- Support custom emotion add/remove via UI

## metadata.json Schema
```json
{
  "emotions": [
    {
      "name": "neutral",
      "name_th": "ปกติ",
      "audio_file": "neutral.wav",
      "ref_text": "ฉันจะอ่านข้อความนี้ให้คุณฟัง",
      "description": "Normal, balanced speaking tone",
      "is_builtin": true
    }
  ]
}
```

## Default Emotions

| name | name_th | description |
|------|---------|-------------|
| neutral | ปกติ | น้ำเสียงปกติ สมดุล |
| happy | มีความสุข | สดใส ร่าเริง |
| sad | เศร้า | เบาเศร้า อ่อนโยน |
| angry | โกรธ | หนักแน่น รุนแรง |
| excited | ตื่นเต้น | พลังสูง กระตือรือร้น |
| calm | สงบ | ช้า สงบ ผ่อนคลาย |

## EmotionManager Design
- `__init__()` - create presets dir, load metadata
- `get_emotion_names()` → list of English names
- `get_emotion_display_names()` → list of "ไทย (english)" format
- `get_emotion_preset(name)` → `{audio_path, ref_text, name, name_th}` or None
- `add_custom_emotion(name, name_th, audio_file_path, ref_text, description)` → bool
- `remove_custom_emotion(name)` → bool (only non-builtin)
- `has_audio_files()` → bool (check if any preset has actual audio)
