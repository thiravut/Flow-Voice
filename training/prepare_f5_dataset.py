"""
prepare_f5_dataset.py

Converts Porjai parquet dataset -> F5-TTS training format.

Output layout:
    {output}/
    |-- wavs/               WAV 24kHz mono files
    |-- metadata.csv        audio_file|text_ipa  (no header)
    +-- vocab.txt           copied from pretrained model

CLI:
    python training/prepare_f5_dataset.py \
        --input dataset/ \
        --output training_data/f5/ \
        --max-samples 1000
"""

import argparse
import os
import shutil
import sys
import warnings
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24000
MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
MIN_TEXT_CHARS = 3

VOCAB_TXT_PATH = None  # auto-detected from HF cache or f5_tts package

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _import_th_to_g2p():
    """Import th_to_g2p lazily so the script fails fast with a clear message."""
    try:
        from f5_tts_th.utils_infer import th_to_g2p
        return th_to_g2p
    except ImportError as exc:
        print(
            "ERROR: Cannot import f5_tts_th.utils_infer.\n"
            "Install it with: pip install f5-tts-th\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def mp3_bytes_to_numpy(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode MP3 bytes -> (float32 numpy array [-1,1], sample_rate)."""
    segment = AudioSegment.from_mp3(BytesIO(mp3_bytes))
    segment = segment.set_channels(1)  # force mono
    raw = np.array(segment.get_array_of_samples(), dtype=np.float32)
    # Normalise to [-1, 1] based on bit depth
    bit_depth = segment.sample_width * 8
    raw /= float(2 ** (bit_depth - 1))
    return raw, segment.frame_rate


def resample_to_target(audio: np.ndarray, src_sr: int, tgt_sr: int = TARGET_SR) -> np.ndarray:
    """Resample numpy float32 array to target sample rate using torchaudio."""
    if src_sr == tgt_sr:
        return audio
    import torch
    tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    resampled = torchaudio.functional.resample(tensor, orig_freq=src_sr, new_freq=tgt_sr)
    return resampled.squeeze(0).numpy()


def duration_seconds(audio: np.ndarray, sr: int) -> float:
    """Return duration in seconds for a 1-D audio array."""
    return len(audio) / sr


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def find_vocab_txt() -> str | None:
    """Find vocab.txt from f5_tts package or HF cache (cross-platform)."""
    # 1. Try from f5_tts package (works on both Linux/Windows)
    try:
        from importlib.resources import files
        pkg_vocab = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
        if os.path.isfile(pkg_vocab):
            return pkg_vocab
    except Exception:
        pass

    # 2. Scan HF hub cache for the model
    hub_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_dir = os.path.join(hub_root, "models--VIZINTZOR--F5-TTS-TH-V2")
    if not os.path.isdir(model_dir):
        return None
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for snapshot in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, snapshot, "vocab.txt")
        if os.path.isfile(candidate):
            return candidate
    return None


def copy_vocab(output_dir: str) -> None:
    """Copy vocab.txt to output_dir. Warns but does not abort if not found."""
    dest = os.path.join(output_dir, "vocab.txt")
    vocab_path = find_vocab_txt()
    if vocab_path is None:
        warnings.warn(
            "vocab.txt not found in HF cache. "
            "Download the model first or copy vocab.txt manually to: " + dest
        )
        return
    shutil.copy2(vocab_path, dest)
    print(f"Copied vocab.txt from {vocab_path}")


# ---------------------------------------------------------------------------
# Parquet processing
# ---------------------------------------------------------------------------

def get_parquet_files(input_dir: str) -> list[str]:
    """Return sorted list of .parquet file paths inside input_dir."""
    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith(".parquet")
    ]
    if not files:
        print(f"ERROR: No .parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def process_parquet_file(
    parquet_path: str,
    wavs_dir: str,
    th_to_g2p,
    global_index: int,
    max_samples: int | None,
    csv_lines: list[str],
    stats: dict,
) -> int:
    """
    Process one parquet file. Mutates csv_lines and stats in-place.
    Returns updated global_index.
    """
    table = pq.read_table(parquet_path, columns=["audio", "sentence", "utterance"])
    num_rows = len(table)

    pbar = tqdm(
        range(num_rows),
        desc=os.path.basename(parquet_path),
        unit="row",
        leave=False,
    )

    for row_idx in pbar:
        if max_samples is not None and global_index >= max_samples:
            break

        stats["total"] += 1

        # --- Extract fields ---
        sentence = table["sentence"][row_idx].as_py()
        audio_col = table["audio"][row_idx].as_py()

        # --- Filter: sentence too short ---
        if not sentence or len(sentence.strip()) < MIN_TEXT_CHARS:
            stats["skipped_text"] += 1
            continue

        # --- Extract MP3 bytes ---
        if isinstance(audio_col, dict):
            mp3_bytes = audio_col.get("bytes")
        elif isinstance(audio_col, bytes):
            mp3_bytes = audio_col
        else:
            stats["skipped_audio_error"] += 1
            continue

        if not mp3_bytes:
            stats["skipped_audio_error"] += 1
            continue

        # --- Decode MP3 -> numpy ---
        try:
            audio_np, src_sr = mp3_bytes_to_numpy(mp3_bytes)
        except Exception:
            stats["skipped_audio_error"] += 1
            continue

        # --- Filter: duration ---
        dur = duration_seconds(audio_np, src_sr)
        if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
            stats["skipped_duration"] += 1
            continue

        # --- Resample to 24kHz ---
        try:
            audio_24k = resample_to_target(audio_np, src_sr, TARGET_SR)
        except Exception:
            stats["skipped_audio_error"] += 1
            continue

        # --- Convert Thai text -> IPA ---
        try:
            ipa_text = th_to_g2p(sentence.strip())
        except Exception:
            stats["skipped_g2p"] += 1
            continue

        # th_to_g2p always appends "." -- strip trailing whitespace for safety
        ipa_text = ipa_text.strip()
        if not ipa_text:
            stats["skipped_g2p"] += 1
            continue

        # --- Write WAV ---
        filename = f"porjai_{global_index:08d}.wav"
        wav_path = os.path.join(wavs_dir, filename)
        sf.write(wav_path, audio_24k, TARGET_SR, subtype="PCM_16")

        # --- Append CSV line ---
        # Format: filename|text_ipa  (filename only, train_f5.py prepends wavs/)
        csv_lines.append(f"{filename}|{ipa_text}")

        global_index += 1
        stats["written"] += 1
        pbar.set_postfix(written=global_index)

    return global_index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Porjai parquet dataset to F5-TTS training format."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Directory containing .parquet files (e.g. dataset/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory (e.g. training_data/f5/)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Stop after writing N samples (useful for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    # --- Validate input ---
    if not os.path.isdir(input_dir):
        print(f"ERROR: --input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")

    # --- Create output dirs ---
    os.makedirs(wavs_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    if args.max_samples is not None:
        print(f"Max samples: {args.max_samples}")

    # --- Lazy-import G2P ---
    th_to_g2p = _import_th_to_g2p()
    print("G2P (th_to_g2p) loaded.")

    # --- Copy vocab.txt ---
    copy_vocab(output_dir)

    # --- Process parquet files ---
    stats = {
        "total": 0,
        "written": 0,
        "skipped_text": 0,
        "skipped_duration": 0,
        "skipped_audio_error": 0,
        "skipped_g2p": 0,
    }
    csv_lines: list[str] = []
    global_index = 0

    file_pbar = tqdm(parquet_files, desc="Parquet files", unit="file")
    for parquet_path in file_pbar:
        if args.max_samples is not None and global_index >= args.max_samples:
            print(f"\nReached --max-samples limit ({args.max_samples}). Stopping.")
            break
        global_index = process_parquet_file(
            parquet_path=parquet_path,
            wavs_dir=wavs_dir,
            th_to_g2p=th_to_g2p,
            global_index=global_index,
            max_samples=args.max_samples,
            csv_lines=csv_lines,
            stats=stats,
        )
        file_pbar.set_postfix(written=global_index)

    # --- Flush metadata.csv (no header, pipe-separated) ---
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
        if csv_lines:
            f.write("\n")

    # --- Summary ---
    print("\n--- Preparation complete ---")
    print(f"  Parquet rows processed : {stats['total']}")
    print(f"  WAV files written      : {stats['written']}")
    print(f"  Skipped (text short)   : {stats['skipped_text']}")
    print(f"  Skipped (duration)     : {stats['skipped_duration']}")
    print(f"  Skipped (audio error)  : {stats['skipped_audio_error']}")
    print(f"  Skipped (G2P error)    : {stats['skipped_g2p']}")
    print(f"\n  metadata.csv           : {metadata_path}")
    print(f"  WAVs directory         : {wavs_dir}")
    vocab_dest = os.path.join(output_dir, "vocab.txt")
    if os.path.isfile(vocab_dest):
        print(f"  vocab.txt              : {vocab_dest}")
    else:
        print("  vocab.txt              : NOT FOUND -- copy manually before training")


if __name__ == "__main__":
    main()
