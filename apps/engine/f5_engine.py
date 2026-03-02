"""
f5_engine.py -- F5-TTS-TH-V2 wrapper for Thai language synthesis.

Uses f5_tts_th.tts.TTS which handles chunking internally.
Patches torchaudio.load to use soundfile backend (avoids torchcodec issues on Windows).
Patches th_to_g2p to improve Thai word segmentation (reduces unnatural pauses).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import (
    CFG_STRENGTH,
    INFERENCE_STEPS,
    MAX_CHARS_CHUNK,
    NOISE_REDUCTION,
    OUTPUT_SAMPLE_RATE,
    SPEED,
)
from engine.audio_processor import apply_noise_reduction, trim_leading_silence
from engine.base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch torchaudio.load to avoid torchcodec dependency on Windows.
# f5_tts_th internally calls torchaudio.load() which fails without FFmpeg DLLs.
# We replace it with a soundfile-based loader that returns the same format.
# ---------------------------------------------------------------------------
_original_torchaudio_load = torchaudio.load


def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Load audio using soundfile, returning (tensor, sample_rate) like torchaudio."""
    try:
        return _original_torchaudio_load(filepath, *args, **kwargs)
    except (RuntimeError, OSError):
        audio, sr = sf.read(filepath, dtype="float32")
        tensor = torch.from_numpy(audio)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)  # (samples,) -> (1, samples)
        else:
            tensor = tensor.T  # (samples, channels) -> (channels, samples)
        return tensor, sr


torchaudio.load = _patched_torchaudio_load

# ---------------------------------------------------------------------------
# Patch th_to_g2p to improve Thai word segmentation.
# The default pythainlp dictionary is missing some compound words (e.g.
# "กระเพรา"), causing them to be split into sub-words.  Each word boundary
# becomes a space in IPA that the model interprets as a pause, resulting in
# unnatural word spacing.  We enhance the dictionary with missing compounds.
# ---------------------------------------------------------------------------
_EXTRA_THAI_WORDS = {
    # กระ- prefix compounds missing from pythainlp dict
    "กระเพรา", "กะเพรา",
}


def _patch_th_to_g2p() -> None:
    """Monkey-patch f5_tts_th.utils_infer.th_to_g2p with improved tokeniser."""
    try:
        from pythainlp.corpus import thai_words
        from pythainlp.tokenize import Tokenizer

        from f5_tts_th import utils_infer
        from f5_tts_th.THG2P import g2p
        from f5_tts_th.normalize import normalize_text

        merged_dict = thai_words() | _EXTRA_THAI_WORDS
        _custom_tok = Tokenizer(custom_dict=merged_dict, engine="newmm")

        def _improved_th_to_g2p(text: str) -> str:
            text = normalize_text(text)
            words = _custom_tok.word_tokenize(text)
            ipa = g2p(words, "ipa")
            if ipa.endswith("."):
                return ipa.replace("  ", " ")
            return ipa.replace("  ", " ") + "."

        utils_infer.th_to_g2p = _improved_th_to_g2p
        logger.info("F5Engine: patched th_to_g2p with improved tokeniser (%d extra words).",
                     len(_EXTRA_THAI_WORDS))
    except Exception as exc:
        logger.warning("F5Engine: could not patch th_to_g2p (using default): %s", exc)


_patch_th_to_g2p()


class F5Engine(BaseTTSEngine):
    """TTS engine backed by F5-TTS-TH-V2 for Thai language synthesis."""

    def __init__(self, model_version: str = "v2", custom_checkpoint: str | None = None) -> None:
        super().__init__()
        self._model_version = model_version
        self._custom_checkpoint = custom_checkpoint

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load F5-TTS-TH model onto the GPU (auto-downloads on first run)."""
        if self._loaded:
            logger.debug("F5Engine: model already loaded, skipping.")
            return

        logger.info("F5Engine: loading F5-TTS-TH-%s model...", self._model_version)
        try:
            if self._custom_checkpoint:
                self._load_finetuned()
            else:
                self._load_pretrained()

            self._loaded = True
            logger.info("F5Engine: model loaded successfully.")
        except ImportError as exc:
            raise RuntimeError(
                "f5-tts-th package is not installed. "
                "Run: pip install f5-tts-th"
            ) from exc
        except Exception as exc:
            self._model = None
            self._loaded = False
            logger.error("F5Engine: failed to load model: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to load F5-TTS-TH model: {exc}") from exc

    def _load_pretrained(self) -> None:
        """Load the standard pretrained model."""
        from f5_tts_th.tts import TTS
        self._model = TTS(model=self._model_version)

    def _load_finetuned(self) -> None:
        """Load a fine-tuned checkpoint while reusing the TTS wrapper."""
        from f5_tts_th.tts import TTS
        from f5_tts_th.utils_infer import load_model, load_vocoder

        logger.info("F5Engine: loading fine-tuned checkpoint: %s", self._custom_checkpoint)

        # Find vocab.txt next to the checkpoint, or in training_data/f5/
        ckpt_dir = os.path.dirname(os.path.abspath(self._custom_checkpoint))
        # project root = apps/../ = voice/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vocab_candidates = [
            os.path.join(ckpt_dir, "vocab.txt"),
            os.path.join(project_root, "training_data", "f5", "vocab.txt"),
        ]
        vocab_path = None
        for v in vocab_candidates:
            if os.path.isfile(v):
                vocab_path = v
                break

        if vocab_path is None:
            raise FileNotFoundError(
                f"vocab.txt not found. Looked in: {vocab_candidates}"
            )

        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, text_mask_padding=True,
            conv_layers=4, pe_attn_head=None,
        )
        f5_model = load_model(
            model_cfg,
            self._custom_checkpoint,
            mel_spec_type="vocos",
            vocab_file=vocab_path,
        )

        # Create a TTS wrapper and replace its model
        tts = TTS.__new__(TTS)
        tts.model_type = self._model_version
        tts.vocoder_name = "vocos"
        tts.hf_cache_dir = None
        tts.f5_model = f5_model
        tts.vocoder = load_vocoder("vocos")
        self._model = tts

    def unload_model(self) -> None:
        """Delete the model and free GPU memory."""
        if not self._loaded:
            return

        logger.info("F5Engine: unloading model and freeing VRAM...")
        del self._model
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("F5Engine: model unloaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, ref_audio: str, gen_text: str, **kwargs: Any) -> str:
        """Synthesise Thai speech and return the path of the saved WAV file.

        Args:
            ref_audio: Path to reference audio (3-15 s).
            gen_text: Thai text to synthesise.
            **kwargs:
                ref_text (str): Transcription of the reference audio (optional).
                steps (int): Diffusion steps (8-64, default 32).
                cfg_strength (float): CFG guidance (0.5-10.0, default 2.0).
                speed (float): Speed multiplier (0.3-3.0, default 1.0).
                max_chars_chunk (int): Max chars per chunk (default 100).
                noise_reduction (bool): Apply noise reduction (default False).
        """
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

        gen_text = gen_text.strip()
        if not gen_text:
            raise ValueError("gen_text must not be empty.")

        ref_text: str = kwargs.get("ref_text", "")
        steps: int = int(kwargs.get("steps", INFERENCE_STEPS))
        cfg_strength: float = float(kwargs.get("cfg_strength", CFG_STRENGTH))
        speed: float = float(kwargs.get("speed", SPEED))
        max_chars_chunk: int = int(kwargs.get("max_chars_chunk", MAX_CHARS_CHUNK))
        use_noise_reduction: bool = bool(kwargs.get("noise_reduction", NOISE_REDUCTION))

        with self._with_lock():
            if not self._loaded:
                self.load_model()

            try:
                logger.info(
                    "F5Engine: generating | steps=%d cfg=%.2f speed=%.2f max_chars=%d",
                    steps, cfg_strength, speed, max_chars_chunk,
                )
                # F5-TTS handles chunking internally via max_chars param
                wav = self._model.infer(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    step=steps,
                    speed=speed,
                    cfg=cfg_strength,
                    max_chars=max_chars_chunk,
                )
            except Exception as exc:
                logger.error("F5Engine: inference failed: %s", exc, exc_info=True)
                raise RuntimeError(f"Speech generation failed: {exc}") from exc

        # Post-processing outside lock
        audio_array = np.array(wav, dtype=np.float32) if not isinstance(wav, np.ndarray) else wav
        output_path = self._save_to_history(audio_array, OUTPUT_SAMPLE_RATE)

        # Trim variable leading silence for consistent playback
        try:
            trim_leading_silence(output_path)
        except Exception as exc:
            logger.warning("F5Engine: trim silence failed (skipped): %s", exc)

        if use_noise_reduction:
            try:
                output_path = apply_noise_reduction(output_path, output_path)
            except Exception as exc:
                logger.warning("F5Engine: noise reduction failed (skipped): %s", exc)

        return output_path

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> list[str]:
        return ["th"]
