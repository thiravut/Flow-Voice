# Product Requirements Document (PRD)
# Multi-Language TTS Voice Cloning Web UI

## 1. Overview

ระบบสร้างเสียงพูดหลายภาษาผ่าน Web UI รองรับการโคลนเสียงจากไฟล์เสียงต้นฉบับ ปรับจูนพารามิเตอร์เสียง และเลือกอารมณ์การพูดได้

ใช้ **2 TTS engines** ร่วมกัน:
- **F5-TTS-TH-V2** — ภาษาไทย (คุณภาพสูงสุดสำหรับไทย)
- **Coqui XTTS v2** — 17 ภาษา (EN, JA, ZH, KO, FR, DE, ES, IT, PT, PL, TR, RU, NL, CS, AR, HU, HI)

ระบบจะ route ไปยัง engine ที่เหมาะสมโดยอัตโนมัติตามภาษาที่เลือก

### 1.1 Problem Statement
การสร้างเสียงพูดที่เป็นธรรมชาติและสามารถโคลนเสียงได้ยังต้องใช้ความรู้ทางเทคนิคสูง ผู้ใช้ทั่วไปไม่สามารถเข้าถึงเทคโนโลยีนี้ได้ง่าย โดยเฉพาะ TTS ภาษาไทยที่มีตัวเลือกน้อย

### 1.2 Solution
Web UI ที่ใช้งานง่าย รันบนเครื่องผู้ใช้เอง (local) รองรับ:
- อัพโหลดเสียงต้นฉบับเพื่อโคลนเสียง
- ปรับจูนคุณภาพและความเร็วเสียง
- เลือกอารมณ์การพูดผ่านระบบ emotion presets
- เลือกภาษาที่ต้องการสร้างเสียง (auto-route ไปยัง engine ที่เหมาะสม)

### 1.3 Target Users
- ผู้ใช้งานส่วนตัวที่ต้องการสร้างเสียงพูดหลายภาษา
- Content creators ที่ต้องการเสียงพากย์
- นักพัฒนาที่ต้องการทดสอบ TTS

---

## 2. Tech Stack

| Component | Technology |
|-----------|------------|
| TTS Model (Thai) | F5-TTS-TH-V2 (`f5-tts-th`) — zero-shot voice cloning, phoneme-based |
| TTS Model (Multi-lang) | Coqui XTTS v2 (`coqui-tts`) — 17 languages, cross-lingual cloning |
| Web UI | Gradio >= 5.0 |
| Audio Processing | soundfile, pydub, librosa, noisereduce |
| Runtime | Python >= 3.10, PyTorch >= 2.2 (CUDA recommended) |
| Output Format | WAV 24kHz |

### 2.1 Engine Comparison

| | F5-TTS-TH-V2 | Coqui XTTS v2 |
|---|---|---|
| **ภาษา** | Thai only | 17 ภาษา |
| **Voice Cloning** | 3-15s ref audio + ref_text | 3-6s ref audio (ไม่ต้อง ref_text) |
| **VRAM** | ~2-4 GB | ~2-4 GB |
| **Output** | WAV 24kHz | WAV 24kHz |
| **Architecture** | Diffusion Transformer + Flow Matching | GPT + VQ-VAE |
| **License** | CC-BY-NC | MPL-2.0 |
| **Routing** | เมื่อเลือกภาษา = Thai | เมื่อเลือกภาษาอื่น |

---

## 3. Features

### 3.1 Voice Cloning (Core Feature)

**Description**: ผู้ใช้อัพโหลดไฟล์เสียงต้นฉบับ 3-15 วินาที ระบบจะโคลนเสียงและสร้างเสียงพูดใหม่ตามข้อความและภาษาที่เลือก

**Acceptance Criteria**:
- [ ] รองรับไฟล์เสียง WAV, MP3, OGG, M4A, FLAC
- [ ] รองรับอัพโหลดไฟล์หรือบันทึกเสียงจากไมโครโฟน
- [ ] แสดงข้อมูลไฟล์เสียง (duration, format, sample rate)
- [ ] ตรวจสอบความยาวเสียง (3-15 วินาที) พร้อมแจ้งเตือน
- [ ] ช่อง ref_text เป็น optional สำหรับ F5-TTS (ไม่จำเป็นสำหรับ XTTS)
- [ ] สร้างเสียงพูดจากข้อความตามภาษาที่เลือก
- [ ] เล่นเสียงผลลัพธ์ได้ทันทีบน UI
- [ ] ดาวน์โหลดไฟล์เสียงผลลัพธ์ได้

### 3.2 Language Selection

**Description**: เลือกภาษาสำหรับสร้างเสียง ระบบจะ route ไปยัง TTS engine ที่เหมาะสมโดยอัตโนมัติ

**Supported Languages**:

| Language | Code | Engine |
|----------|------|--------|
| ไทย | th | F5-TTS-TH-V2 |
| English | en | Coqui XTTS v2 |
| 日本語 | ja | Coqui XTTS v2 |
| 中文 | zh | Coqui XTTS v2 |
| 한국어 | ko | Coqui XTTS v2 |
| Français | fr | Coqui XTTS v2 |
| Deutsch | de | Coqui XTTS v2 |
| Español | es | Coqui XTTS v2 |
| Italiano | it | Coqui XTTS v2 |
| Português | pt | Coqui XTTS v2 |
| Русский | ru | Coqui XTTS v2 |
| العربية | ar | Coqui XTTS v2 |
| हिन्दी | hi | Coqui XTTS v2 |
| Polski | pl | Coqui XTTS v2 |
| Türkçe | tr | Coqui XTTS v2 |
| Nederlands | nl | Coqui XTTS v2 |
| Čeština | cs | Coqui XTTS v2 |
| Magyar | hu | Coqui XTTS v2 |

**Acceptance Criteria**:
- [ ] Dropdown เลือกภาษา (default: Thai)
- [ ] แสดงชื่อ engine ที่ใช้ตามภาษาที่เลือก
- [ ] ref_text field ซ่อนอัตโนมัติเมื่อใช้ XTTS (ไม่จำเป็น)
- [ ] Parameter sliders ปรับตาม engine (F5 มี step/cfg, XTTS ไม่มี)
- [ ] Cross-lingual cloning: ใช้ ref audio ภาษาหนึ่ง สร้างเสียงอีกภาษาได้

### 3.3 Voice Tuning

**Description**: ปรับจูนพารามิเตอร์การสร้างเสียงเพื่อควบคุมคุณภาพและลักษณะเสียง

**Parameters**:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Inference Steps | 8 - 64 | 32 | จำนวนขั้นตอน inference (สูง = คุณภาพดี, ช้ากว่า) |
| CFG Strength | 0.5 - 10.0 | 2.0 | ความเข้มของ guidance (สูง = ใกล้เสียงต้นฉบับมากขึ้น) |
| Speed | 0.3 - 3.0 | 1.0 | ความเร็วการพูด (1.0 = ปกติ) |
| Max Chars/Chunk | 50 - 300 | 100 | ขนาดการแบ่งข้อความยาว |
| Noise Reduction | on/off | off | ลดเสียงรบกวนในผลลัพธ์ |

**Acceptance Criteria**:
- [ ] แต่ละ parameter มี slider พร้อมคำอธิบาย
- [ ] แสดง recommended values (เช่น Steps: 16=เร็ว, 32=สมดุล, 64=คุณภาพสูงสุด)
- [ ] Parameter validation (ไม่ให้เกิน min/max)

### 3.4 Emotion Control

**Description**: ควบคุมอารมณ์เสียงผ่านระบบ Emotion Presets โดยใช้ reference audio ที่บันทึกด้วยอารมณ์ต่างๆ เป็นตัวกำหนด (F5-TTS โคลนทั้งเสียงและอารมณ์/สไตล์การพูดจาก reference audio)

**Default Emotion Presets** (6 อารมณ์):

| Emotion | Thai | Description |
|---------|------|-------------|
| neutral | ปกติ | น้ำเสียงปกติ สมดุล |
| happy | มีความสุข | สดใส ร่าเริง |
| sad | เศร้า | เบาเศร้า อ่อนโยน |
| angry | โกรธ | หนักแน่น รุนแรง |
| excited | ตื่นเต้น | พลังสูง กระตือรือร้น |
| calm | สงบ | ช้า สงบ ผ่อนคลาย |

**Acceptance Criteria**:
- [ ] Dropdown เลือก emotion preset
- [ ] Preview เสียง preset ก่อนใช้งาน
- [ ] แสดง ref_text ของ preset
- [ ] Toggle สลับระหว่างโหมด "Upload Audio" กับ "Emotion Preset"
- [ ] เพิ่ม custom emotion preset ได้ (ชื่อ + audio + ref_text)
- [ ] ลบ custom emotion preset ได้
- [ ] ตาราง overview ของ presets ทั้งหมด

**Note**: ระบบจัดส่งพร้อม metadata ของ 6 อารมณ์ แต่ไม่มีไฟล์เสียง ผู้ใช้ต้องบันทึกหรืออัพโหลดเสียง reference เอง

### 3.5 Generation History

**Description**: บันทึกประวัติเสียงที่สร้างทั้งหมดไว้ในโฟลเดอร์ history/

**Acceptance Criteria**:
- [ ] บันทึกไฟล์ WAV อัตโนมัติทุกครั้งที่สร้างเสียง
- [ ] ตั้งชื่อไฟล์ด้วย timestamp + hash (เช่น `20260225_143022_a1b2c3d4.wav`)
- [ ] แสดงรายการประวัติ (ชื่อไฟล์, วันที่, ขนาด)
- [ ] เล่นเสียงจากประวัติได้
- [ ] จำกัดจำนวนสูงสุด 50 รายการ

---

## 4. Architecture

### 4.1 Project Structure

```
voice/
├── app.py                        # Entry point
├── config.py                     # Constants & default values
├── requirements.txt
├── docs/
│   └── PRD.md
├── engine/
│   ├── __init__.py
│   ├── base_engine.py            # Abstract base class for TTS engines
│   ├── f5_engine.py              # F5-TTS-TH-V2 wrapper (Thai)
│   ├── coqui_engine.py           # Coqui XTTS v2 wrapper (multi-lang)
│   ├── engine_router.py          # Routes to correct engine by language
│   └── audio_processor.py        # Audio validation & processing (shared)
├── ui/
│   ├── __init__.py
│   ├── app_ui.py                 # Gradio UI (3 tabs)
│   └── custom.css
├── emotions/
│   ├── __init__.py
│   ├── emotion_manager.py        # Emotion preset CRUD
│   └── presets/
│       └── metadata.json
├── history/                      # Generated audio (auto-created)
└── uploads/                      # User uploads (auto-created)
```

### 4.2 Component Diagram

```
┌─────────────────────────────────────────────────────┐
│                     app.py                          │
│                (Entry Point)                        │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                 ui/app_ui.py                        │
│              (Gradio Web UI)                        │
│                                                     │
│  ┌───────────────┐ ┌──────────┐ ┌──────────────┐    │
│  │  Tab 1:       │ │  Tab 2:  │ │   Tab 3:     │    │
│  │  Generate     │ │ Emotions │ │   History    │    │
│  │  + Lang Select│ │          │ │              │    │
│  └───────┬───────┘ └────┬─────┘ └──────────────┘    │
└──────────┼──────────────┼───────────────────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────────────┐
    │ engine/     │  │ emotions/         │
    │ engine_     │  │ emotion_manager   │
    │ router      │  │ presets/metadata  │
    └──┬──────┬───┘  └───────────────────┘
       │      │
       │      │  ┌────────────────────┐
       │      └─►│ coqui_engine.py    │
       │         │ (Coqui XTTS v2)   │
       │         │ 17 languages      │
       │         └────────┬───────────┘
       │                  │
       │  ┌──────────┐    │  ┌────────────┐
       └─►│ f5_engine│    └─►│ Coqui TTS  │
          │ (F5-TTS) │       │ (External) │
          │ Thai     │       └────────────┘
          └────┬─────┘
               │
          ┌────▼────────┐
          │ f5_tts_th   │
          │ (External)  │
          └─────────────┘
```

### 4.3 Engine Router Logic

```
User selects language
        │
        ▼
  ┌─ language == "th" ──► Load F5Engine (if not loaded)
  │                       Unload CoquiEngine (if VRAM tight)
  │                       F5Engine.generate(ref_audio, ref_text, gen_text, ...)
  │
  └─ language != "th" ──► Load CoquiEngine (if not loaded)
                          Unload F5Engine (if VRAM tight)
                          CoquiEngine.generate(ref_audio, gen_text, language, ...)
```

### 4.4 Key Design Decisions

| Decision | Why |
|----------|-----|
| **Multi-engine + Router** | F5-TTS ดีที่สุดสำหรับไทย, XTTS v2 ครอบคลุม 17 ภาษาอื่น |
| **VRAM swap strategy** | แต่ละ model ใช้ ~2-4GB, swap ออกเมื่อไม่ใช้เพื่อประหยัด VRAM |
| **Abstract base class** | ให้ทุก engine มี interface เดียวกัน, เพิ่ม engine ใหม่ง่าย |
| **Thread lock** | Gradio ใช้ threading, ป้องกัน CUDA OOM |
| **Emotion ผ่าน reference audio** | ทั้ง F5-TTS และ XTTS โคลน style จาก ref audio เหมือนกัน |
| **Gradio** | เป็น dependency ของทั้ง 2 engines, มี audio components พร้อมใช้ |
| **Noise reduction optional** | Output สะอาดอยู่แล้ว, เปิดใช้เฉพาะเมื่อต้องการ |

---

## 5. UI Mockup

```
╔══════════════════════════════════════════════════════════════╗
║  Multi-Language TTS Voice Cloning                           ║
║  Voice cloning with F5-TTS (Thai) & XTTS v2 (17 languages) ║
╠══════════════════════════════════════════════════════════════╣
║  [Generate Speech]  [Emotion Presets]  [History]             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Language: [ th - ไทย          ▼]  Engine: F5-TTS-TH-V2     ║
║                                                              ║
║  ┌─── Reference Voice ──────┐  ┌─── Generation Settings ──┐ ║
║  │                          │  │                           │ ║
║  │ Mode: (●)Upload (○)Emot. │  │ Steps:    ────●───── 32  │ ║
║  │                          │  │ CFG:      ──●─────── 2.0 │ ║
║  │ ┌──────────────────────┐ │  │ Speed:    ────●───── 1.0 │ ║
║  │ │  Upload Audio        │ │  │ Max Chars: ───●──── 100  │ ║
║  │ │  (drag & drop / mic) │ │  │                           │ ║
║  │ └──────────────────────┘ │  │ [ ] Noise Reduction       │ ║
║  │                          │  │                           │ ║
║  │ Ref Text: [............] │  │ (params adapt per engine) │ ║
║  │ Info: 5.2s | WAV | 24kHz│  └───────────────────────────┘ ║
║  └──────────────────────────┘                                ║
║                                                              ║
║  Text to Generate:                                           ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ พิมพ์ข้อความที่ต้องการให้อ่าน...                            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  [ ████████ Generate Speech ████████ ]                       ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  ▶ Generated Speech  ───●────────────── 00:05 / 00:12   │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║  Status: Generated successfully. Saved to history.           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 6. Implementation Phases

### Phase 1: Foundation
- สร้าง directory structure, `requirements.txt`, `config.py`
- Install dependencies (`f5-tts-th`, `coqui-tts`, `gradio`, `noisereduce`)

### Phase 2: Engine Layer
- `engine/base_engine.py` - Abstract base class (interface contract)
- `engine/f5_engine.py` - F5-TTS-TH wrapper (Thai)
- `engine/coqui_engine.py` - Coqui XTTS v2 wrapper (multi-lang)
- `engine/engine_router.py` - Language-based routing + VRAM management
- `engine/audio_processor.py` - Shared audio utilities

### Phase 3: Emotion System
- `emotions/presets/metadata.json` - 6 default emotions
- `emotions/emotion_manager.py` - Preset management

### Phase 4: Web UI
- `ui/custom.css` - Styling
- `ui/app_ui.py` - Gradio interface (3 tabs) + language selector

### Phase 5: Integration & Test
- `app.py` - Entry point
- End-to-end testing (Thai + multi-lang)

---

## 7. Verification Plan

| Test | Expected Result |
|------|----------------|
| `python app.py` | Browser opens at http://localhost:7860 |
| เลือก Thai + Upload WAV + Generate | เสียงพูดภาษาไทยจาก F5-TTS |
| เลือก English + Upload WAV + Generate | เสียงพูดภาษาอังกฤษจาก XTTS v2 |
| เลือก Japanese + Generate | เสียงพูดภาษาญี่ปุ่นจาก XTTS v2 |
| สลับภาษา Thai → English → Thai | Engine swap สำเร็จ, ไม่ crash |
| ปรับ Steps (Thai mode) | Steps slider ปรากฏ, คุณภาพเปลี่ยนตาม |
| เลือกภาษาอื่น | ref_text field ซ่อน (XTTS ไม่ต้องใช้) |
| เพิ่ม custom emotion + Generate | เสียงมีอารมณ์ตาม reference |
| สร้างเสียงหลายครั้ง | ไฟล์ปรากฏใน History tab |
| Upload ไฟล์ < 3 วินาที | แจ้งเตือน "Audio too short" |
| ไม่ใส่ข้อความ + Generate | แจ้งเตือน "Please enter text" |
| Cross-lingual: ref=ไทย, gen=EN | โคลนเสียงไทยพูดภาษาอังกฤษ |

---

## 8. Constraints & Limitations

- **GPU required**: การสร้างเสียงต้องใช้ GPU (CUDA) เพื่อประสิทธิภาพที่ดี
- **VRAM**: แต่ละ engine ใช้ ~2-4GB, ระบบ swap model เพื่อประหยัด VRAM (GPU >= 8GB แนะนำ)
- **First-run download**: ครั้งแรกจะดาวน์โหลดโมเดลจาก HuggingFace (F5: ~1-2GB, XTTS: ~2GB)
- **Single user**: ออกแบบสำหรับใช้งานคนเดียว (local), ไม่รองรับ concurrent users
- **Emotion quality**: คุณภาพอารมณ์ขึ้นอยู่กับคุณภาพ reference audio ที่ผู้ใช้บันทึก
- **FFmpeg required**: ต้องติดตั้ง FFmpeg สำหรับรองรับไฟล์ MP3/M4A
- **Engine swap latency**: การสลับระหว่าง F5 ↔ XTTS ครั้งแรกอาจใช้เวลา 5-10 วินาที
- **License mixing**: F5-TTS = CC-BY-NC (non-commercial), XTTS = MPL-2.0 (commercial OK)
