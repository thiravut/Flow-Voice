"""
prepare_xtts_dataset.py

Converts Porjai parquet dataset to Coqui XTTS training format.

Output layout:
    <output>/
        wavs/               - WAV files (22050 Hz mono)
        metadata_train.csv  - pipe-delimited, with header
        metadata_eval.csv   - pipe-delimited, with header
        lang.txt            - contains "th"

CSV format (coqui formatter):
    audio_file|text|speaker_name
    wavs/porjai_00000000.wav|ข้อความ|porjai

WARNING: Thai (th) is NOT natively supported by the XTTS v2 tokenizer.
The tokenizer raises NotImplementedError when lang="th" is passed at training
time. Before running XTTS fine-tuning with this dataset you have one of these
options:
  1. Patch the XTTS tokenizer to add Thai character support (recommended).
  2. Pre-romanize the Thai text with a library such as `pythainlp.transliterate`
     and use lang="en" as the base tokenizer. Quality will be lower.
  3. Use a different TTS engine that natively supports Thai (e.g. F5-TTS-TH-V2).
See: https://github.com/coqui-ai/TTS/issues for community patches.
"""

import argparse
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEAKER_NAME = "porjai"
SAMPLE_RATE = 22050
MIN_DURATION_S = 2.0
MAX_DURATION_S = 10.0
TRAIN_RATIO = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mp3_bytes_to_tensor(mp3_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode MP3 bytes to a float32 waveform tensor via pydub.

    Returns (waveform, sample_rate) where waveform shape is (channels, samples).
    """
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    # Convert to raw PCM samples
    samples = audio_segment.get_array_of_samples()
    waveform = torch.tensor(samples, dtype=torch.float32)
    # Reshape to (channels, samples)
    n_channels = audio_segment.channels
    waveform = waveform.reshape(-1, n_channels).T  # (channels, samples)
    # Normalise int range to [-1.0, 1.0]
    max_val = float(2 ** (audio_segment.sample_width * 8 - 1))
    waveform = waveform / max_val
    return waveform, audio_segment.frame_rate


def to_mono_22050(waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Convert waveform to mono 22050 Hz."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if src_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=src_sr, new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)
    return waveform


def duration_seconds(waveform: torch.Tensor, sr: int) -> float:
    """Return audio duration in seconds."""
    return waveform.shape[-1] / sr


def wav_filename(index: int) -> str:
    """Return the wav filename (no directory prefix) for a given index."""
    return f"porjai_{index:08d}.wav"


def collect_parquet_files(input_dir: str) -> list[str]:
    """Return sorted list of .parquet file paths under input_dir."""
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ]
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Porjai parquet dataset to Coqui XTTS format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing .parquet files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the XTTS dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (after filtering).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split shuffle (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    wavs_dir = os.path.join(output_dir, "wavs")

    os.makedirs(wavs_dir, exist_ok=True)

    parquet_files = collect_parquet_files(input_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in: {input_dir}")

    print(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Pass 1: collect all valid rows into memory as (text, wav_path)
    # ------------------------------------------------------------------

    records: list[tuple[str, str]] = []  # (text, relative_wav_path)
    global_index = 0
    skipped_short = 0
    skipped_long = 0
    skipped_errors = 0

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pydict()

        audio_col = df.get("audio", [])
        sentence_col = df.get("sentence", [])
        n_rows = len(audio_col)

        desc = os.path.basename(parquet_path)
        for i in tqdm(range(n_rows), desc=desc, unit="row"):
            # Early-exit if max-samples already reached
            if args.max_samples is not None and len(records) >= args.max_samples:
                break

            try:
                audio_dict = audio_col[i]
                mp3_bytes: bytes = audio_dict["bytes"]
                text: str = str(sentence_col[i]).strip()

                if not text:
                    skipped_errors += 1
                    continue

                waveform, src_sr = mp3_bytes_to_tensor(mp3_bytes)
                waveform = to_mono_22050(waveform, src_sr)

                dur = duration_seconds(waveform, SAMPLE_RATE)
                if dur < MIN_DURATION_S:
                    skipped_short += 1
                    continue
                if dur > MAX_DURATION_S:
                    skipped_long += 1
                    continue

                fname = wav_filename(global_index)
                wav_path = os.path.join(wavs_dir, fname)
                # Use soundfile instead of torchaudio.save to avoid torchcodec/FFmpeg dependency
                audio_np = waveform.squeeze(0).numpy()
                sf.write(wav_path, audio_np, SAMPLE_RATE, subtype="PCM_16")

                relative_path = os.path.join("wavs", fname)
                records.append((text, relative_path))
                global_index += 1

            except Exception as exc:  # noqa: BLE001
                skipped_errors += 1
                print(f"\nWarning: skipped row {i} in {desc} — {exc}")

        if args.max_samples is not None and len(records) >= args.max_samples:
            print(f"Reached --max-samples {args.max_samples}, stopping early.")
            break

    print(
        f"\nProcessed {len(records)} valid samples "
        f"(skipped: {skipped_short} too short, "
        f"{skipped_long} too long, "
        f"{skipped_errors} errors)"
    )

    if not records:
        raise RuntimeError("No valid samples were produced. Check your input data.")

    # ------------------------------------------------------------------
    # Pass 2: shuffle + split 90/10
    # ------------------------------------------------------------------

    random.seed(args.seed)
    random.shuffle(records)

    split_idx = max(1, int(len(records) * TRAIN_RATIO))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    print(f"Train: {len(train_records)} samples | Eval: {len(eval_records)} samples")

    # ------------------------------------------------------------------
    # Pass 3: write CSV files
    # ------------------------------------------------------------------

    def write_csv(path: str, rows: list[tuple[str, str]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("audio_file|text|speaker_name\n")
            for text, rel_wav in rows:
                # Escape any pipe characters in text to avoid format breakage
                safe_text = text.replace("|", " ")
                fh.write(f"{rel_wav}|{safe_text}|{SPEAKER_NAME}\n")

    train_csv = os.path.join(output_dir, "metadata_train.csv")
    eval_csv = os.path.join(output_dir, "metadata_eval.csv")
    write_csv(train_csv, train_records)
    write_csv(eval_csv, eval_records)

    # ------------------------------------------------------------------
    # Pass 4: write lang.txt
    # ------------------------------------------------------------------

    lang_txt = os.path.join(output_dir, "lang.txt")
    with open(lang_txt, "w", encoding="utf-8") as fh:
        fh.write("th\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\nDataset preparation complete.")
    print(f"  WAV files  : {wavs_dir}")
    print(f"  Train CSV  : {train_csv}")
    print(f"  Eval CSV   : {eval_csv}")
    print(f"  lang.txt   : {lang_txt}")
    print(
        "\nWARNING: Thai (th) is NOT natively supported by the XTTS v2 tokenizer."
        " See the module docstring for workarounds before starting fine-tuning."
    )


if __name__ == "__main__":
    main()
