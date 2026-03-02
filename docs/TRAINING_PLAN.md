# Training Pipeline สำหรับ Porjai Dataset (RunPod)

## Context

โปรเจกต์ TTS voice cloning มีแค่ inference pipeline ต้องการเพิ่ม training pipeline เพื่อ fine-tune ทั้ง F5-TTS-TH-V2 และ Coqui XTTS v2 ด้วย Porjai dataset (~335K samples, ~700 ชม.) เพื่อ:
- **Goal A**: ปรับปรุงคุณภาพเสียงไทยโดยรวม (ออกเสียงชัด ธรรมชาติขึ้น)
- **Goal B**: ปรับปรุงความสามารถ voice cloning (clone เสียงจาก reference audio ได้แม่นยำขึ้น)

**Target**: RunPod Pod (SSH) + RTX 4090 (24GB VRAM) — ~$0.4-0.7/hr

## Dataset

```
dataset/train-00000-of-00016.parquet  (x16 files)
- Columns: audio (MP3 bytes, 16kHz), sentence (Thai text), utterance (ID)
- Total: 335,674 rows, ~700 ชม., ~7.2 GB
```

## Files

```
training/
├── setup_runpod.sh             # RunPod setup script (deps + dataset download)
├── requirements_train.txt      # Training dependencies
├── prepare_f5_dataset.py       # Parquet → WAV 24kHz + IPA metadata
├── prepare_xtts_dataset.py     # Parquet → WAV 22050Hz + CSV + lang.txt
├── train_f5.py                 # F5-TTS fine-tuning (accelerate)
├── train_xtts.py               # XTTS v2 fine-tuning (Coqui Trainer)
└── config/
    ├── f5_finetune.yaml        # F5 config (RTX 4090 optimized)
    └── xtts_finetune.yaml      # XTTS config (RTX 4090 optimized)
```

## RunPod Workflow

```bash
# 1. สร้าง Pod: RTX 4090 (24GB), PyTorch template, Volume 100GB+
# 2. SSH เข้า Pod
# 3. Clone repo
git clone <repo-url> /workspace/voice && cd /workspace/voice

# 4. Setup environment + download dataset
bash training/setup_runpod.sh --hf-token YOUR_TOKEN

# 5. Prepare data (ทดสอบ 100 samples ก่อน)
python training/prepare_f5_dataset.py --input /workspace/dataset --output /workspace/training_data/f5/ --max-samples 100

# 6. Start training
python training/train_f5.py --config training/config/f5_finetune.yaml --max-steps 10

# 7. Full training
python training/train_f5.py --config training/config/f5_finetune.yaml

# 8. Download checkpoint กลับเครื่อง local
scp root@<pod-ip>:/workspace/checkpoints/f5/model_XXXXXXX.pt ./checkpoints/

# 9. ตั้ง F5_CUSTOM_CHECKPOINT ใน config.py → ใช้กับ Gradio UI
```

## After Training (Local)

1. Copy checkpoint มาไว้ที่ `checkpoints/f5/model_XXXXXXX.pt`
2. แก้ `apps/config.py`:
   ```python
   F5_CUSTOM_CHECKPOINT = "../checkpoints/f5/model_XXXXXXX.pt"
   ```
3. รัน `python app.py` → เสียงไทยจะใช้โมเดลที่ fine-tune แล้ว
