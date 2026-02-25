"""
audio_processor.py — Shared audio utilities for loading, validation,
text chunking, noise reduction, and concatenation.
"""

import os
import re

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

from config import (
    MAX_AUDIO_DURATION,
    MAX_CHARS_CHUNK,
    MIN_AUDIO_DURATION,
    OUTPUT_SAMPLE_RATE,
    SUPPORTED_FORMATS,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_audio(file_path: str) -> dict:
    """Validate a reference audio file for format and duration constraints.

    Returns a dict with keys: valid, duration, format, sample_rate, error.
    """
    result: dict = {
        "valid": False,
        "duration": 0.0,
        "format": "",
        "sample_rate": 0,
        "error": None,
    }

    # --- format check ---
    ext = os.path.splitext(file_path)[-1].lstrip(".").lower()
    if not ext:
        result["error"] = "File has no extension. Supported formats: " + ", ".join(SUPPORTED_FORMATS)
        return result

    if ext not in SUPPORTED_FORMATS:
        result["error"] = (
            f"Unsupported format '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
        return result

    result["format"] = ext

    # --- existence check ---
    if not os.path.isfile(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    # --- load and analyse ---
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as exc:
        result["error"] = f"Could not read audio file: {exc}"
        return result

    duration = float(librosa.get_duration(y=audio, sr=sr))
    result["duration"] = round(duration, 3)
    result["sample_rate"] = int(sr)

    # --- duration check ---
    if duration < MIN_AUDIO_DURATION:
        result["error"] = (
            f"Audio too short ({duration:.1f}s). "
            f"Minimum is {MIN_AUDIO_DURATION:.0f}s."
        )
        return result

    if duration > MAX_AUDIO_DURATION:
        result["error"] = (
            f"Audio too long ({duration:.1f}s). "
            f"Maximum is {MAX_AUDIO_DURATION:.0f}s."
        )
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


def get_audio_info(file_path: str) -> dict:
    """Return basic audio metadata for UI display.

    Returns a dict with keys: duration, format, sample_rate, error.
    """
    info: dict = {
        "duration": 0.0,
        "format": "",
        "sample_rate": 0,
        "error": None,
    }

    ext = os.path.splitext(file_path)[-1].lstrip(".").lower()
    info["format"] = ext

    if not os.path.isfile(file_path):
        info["error"] = f"File not found: {file_path}"
        return info

    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        info["duration"] = round(float(librosa.get_duration(y=audio, sr=sr)), 3)
        info["sample_rate"] = int(sr)
    except Exception as exc:
        info["error"] = f"Could not read audio file: {exc}"

    return info


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

# Sentence-ending characters covering Latin, Thai, CJK, Arabic, Devanagari.
_SENTENCE_END_PATTERN = re.compile(
    r"(?<=[.!?。！？।۔\n])\s*"   # after sentence-enders (incl. newline)
)

# Whitespace word boundary splitter (for Latin/space-delimited scripts).
_WORD_SPLIT_PATTERN = re.compile(r"\s+")


def chunk_text(text: str, max_chars: int = MAX_CHARS_CHUNK) -> list[str]:
    """Split text into chunks of at most max_chars characters.

    Splitting precedence:
    1. Sentence boundaries (. ! ? newlines, CJK/Thai sentence-enders).
    2. Word/whitespace boundaries for Latin-script languages.
    3. Hard character-count cut when no boundary is available (Thai words).
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # --- pass 1: split on sentence boundaries ---
    sentences = [s.strip() for s in _SENTENCE_END_PATTERN.split(text) if s.strip()]

    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        if not sentence:
            continue

        # Sentence itself is within limit — try to append to buffer.
        if len(sentence) <= max_chars:
            candidate = (buffer + " " + sentence).strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sentence
        else:
            # Sentence exceeds limit — flush buffer, then split sentence further.
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long_segment(sentence, max_chars))

    if buffer:
        chunks.append(buffer)

    return chunks


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    """Split a single long segment by word boundaries, then hard-cut if needed."""
    if len(text) <= max_chars:
        return [text]

    # Try word-boundary split (works for space-delimited languages).
    words = _WORD_SPLIT_PATTERN.split(text)

    # If the text has no spaces (e.g. Thai), words == [text]; fall through to hard cut.
    if len(words) <= 1:
        return _hard_cut(text, max_chars)

    chunks: list[str] = []
    buffer = ""

    for word in words:
        candidate = (buffer + " " + word).strip() if buffer else word
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # Single word longer than limit — hard cut it.
            if len(word) > max_chars:
                chunks.extend(_hard_cut(word, max_chars))
                buffer = ""
            else:
                buffer = word

    if buffer:
        chunks.append(buffer)

    return chunks


def _hard_cut(text: str, max_chars: int) -> list[str]:
    """Cut text into fixed-size slices with no regard for boundaries."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------


def apply_noise_reduction(
    audio_path: str,
    output_path: str | None = None,
) -> str:
    """Apply spectral noise reduction to an audio file and save the result.

    If output_path is None the result is written next to the source file
    with a '_nr' suffix. Returns the output path.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    reduced: np.ndarray = nr.reduce_noise(y=audio, sr=sr)

    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = base + "_nr" + (ext if ext else ".wav")

    sf.write(output_path, reduced, sr, subtype="PCM_16")
    return output_path


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------


def concatenate_audio(
    audio_paths: list[str],
    output_path: str,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> str:
    """Concatenate multiple audio files into a single WAV file.

    All inputs are resampled to sample_rate before joining. Returns output_path.
    """
    if not audio_paths:
        raise ValueError("audio_paths must not be empty.")

    segments: list[np.ndarray] = []

    for path in audio_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sr = librosa.load(path, sr=sample_rate, mono=True)
        segments.append(audio)

    combined: np.ndarray = np.concatenate(segments)
    sf.write(output_path, combined, sample_rate, subtype="PCM_16")
    return output_path
