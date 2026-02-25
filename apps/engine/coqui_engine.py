"""
coqui_engine.py -- Coqui XTTS v2 wrapper for multi-language TTS (17 languages).

Supports cross-lingual voice cloning: reference audio can be in any language
while the synthesised output is in the target language specified via language param.
"""

from __future__ import annotations

import logging
import os

import librosa
import numpy as np
import torch

from config import OUTPUT_SAMPLE_RATE, SPEED, XTTS_MODEL_NAME
from engine.audio_processor import apply_noise_reduction
from engine.base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)

# XTTS v2 native sample rate -- the model always produces audio at 24 kHz.
_XTTS_SAMPLE_RATE: int = 24000

# All language codes supported by XTTS v2 (Thai excluded -- handled by F5).
_SUPPORTED_LANGUAGES: list[str] = [
    "en", "ja", "zh", "ko", "fr", "de", "es", "it",
    "pt", "ru", "ar", "hi", "pl", "tr", "nl", "cs", "hu",
]


class CoquiEngine(BaseTTSEngine):
    """Coqui XTTS v2 TTS engine supporting 17 languages with voice cloning.

    Voice characteristics are cloned from a reference audio file (3-6 s
    recommended). Cross-lingual cloning is supported: the reference audio
    language does not need to match the target generation language.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load XTTS v2 model onto the GPU.

        Downloads the model from HuggingFace on the first run (~1.8 GB).
        Subsequent runs use the cached copy.
        """
        if self._loaded:
            logger.debug("CoquiEngine: model already loaded, skipping.")
            return

        logger.info("CoquiEngine: loading model %r ...", XTTS_MODEL_NAME)
        try:
            from TTS.api import TTS  # imported lazily to avoid overhead at startup

            model = TTS(model_name=XTTS_MODEL_NAME)
            model.to("cuda")
            self._model = model
            self._loaded = True
            logger.info("CoquiEngine: model loaded successfully on GPU.")
        except Exception as exc:
            self._model = None
            self._loaded = False
            logger.error("CoquiEngine: failed to load model: %s", exc, exc_info=True)
            raise RuntimeError(f"CoquiEngine failed to load model: {exc}") from exc

    def unload_model(self) -> None:
        """Move the model off the GPU and release VRAM."""
        if not self._loaded or self._model is None:
            logger.debug("CoquiEngine: model not loaded, nothing to unload.")
            return

        logger.info("CoquiEngine: unloading model from GPU ...")
        try:
            # Move underlying synthesiser sub-models to CPU to free CUDA memory.
            if hasattr(self._model, "synthesizer") and self._model.synthesizer is not None:
                synthesizer = self._model.synthesizer
                if hasattr(synthesizer, "tts_model") and synthesizer.tts_model is not None:
                    synthesizer.tts_model.to("cpu")
                if (
                    hasattr(synthesizer, "vocoder_model")
                    and synthesizer.vocoder_model is not None
                ):
                    synthesizer.vocoder_model.to("cpu")
        except Exception as exc:
            logger.warning("CoquiEngine: error while moving model to CPU: %s", exc)

        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("CoquiEngine: model unloaded and VRAM cache cleared.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, ref_audio: str, gen_text: str, **kwargs) -> str:
        """Generate speech from text using the voice in ref_audio.

        Args:
            ref_audio: Path to reference audio file (3-6 s, clean recording).
            gen_text: Text to synthesise.
            **kwargs:
                language (str, required): BCP-47 language code for the output.
                    Must be one of get_supported_languages(). Cross-lingual
                    cloning is supported -- ref_audio may be in a different
                    language.
                speed (float, optional): Playback speed multiplier.
                    Default is SPEED (1.0). Applied via post-generation
                    resampling since XTTS v2 has no native speed parameter.
                noise_reduction (bool, optional): Apply spectral noise
                    reduction to the output. Default False.

        Returns:
            Absolute path to the generated WAV file inside HISTORY_DIR.

        Raises:
            ValueError: If language is missing or unsupported.
            FileNotFoundError: If ref_audio does not exist.
            RuntimeError: If model loading or inference fails.
        """
        # --- validate language ---
        language: str = kwargs.get("language", "")
        if not language:
            raise ValueError(
                "CoquiEngine.generate() requires the keyword argument: language"
            )
        if language not in _SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Supported: {_SUPPORTED_LANGUAGES}"
            )

        # --- validate reference audio ---
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        # --- optional parameters ---
        speed: float = float(kwargs.get("speed", SPEED))
        noise_reduction: bool = bool(kwargs.get("noise_reduction", False))

        with self._with_lock():
            # Lazy-load if needed (e.g. first call, or after a VRAM swap).
            if not self._loaded:
                self.load_model()

            logger.info(
                "CoquiEngine: generating -- language=%s speed=%.2f noise_reduction=%s",
                language,
                speed,
                noise_reduction,
            )

            try:
                wav: list[float] = self._model.tts(
                    text=gen_text,
                    speaker_wav=ref_audio,
                    language=language,
                )
            except Exception as exc:
                logger.error("CoquiEngine: inference failed: %s", exc, exc_info=True)
                raise RuntimeError(f"CoquiEngine inference failed: {exc}") from exc

        # --- post-processing (outside lock to reduce GPU contention) ---
        audio_array: np.ndarray = np.array(wav, dtype=np.float32)
        sample_rate: int = _XTTS_SAMPLE_RATE

        # Apply speed adjustment via resampling when speed != 1.0.
        if abs(speed - 1.0) > 1e-3:
            audio_array = _apply_speed(audio_array, sample_rate, speed)

        # Resample to project output rate when they differ.
        if sample_rate != OUTPUT_SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=OUTPUT_SAMPLE_RATE
            )
            sample_rate = OUTPUT_SAMPLE_RATE

        output_path: str = self._save_to_history(audio_array, sample_rate)

        if noise_reduction:
            try:
                output_path = apply_noise_reduction(output_path, output_path)
                logger.debug("CoquiEngine: noise reduction applied to %s", output_path)
            except Exception as exc:
                logger.warning(
                    "CoquiEngine: noise reduction failed (output kept as-is): %s", exc
                )

        logger.info("CoquiEngine: output saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Language support
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> list[str]:
        """Return the 17 language codes supported by XTTS v2."""
        return list(_SUPPORTED_LANGUAGES)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_speed(
    audio: np.ndarray,
    sample_rate: int,
    speed: float,
) -> np.ndarray:
    """Adjust playback speed by resampling without preserving pitch.

    Increasing speed (> 1.0) compresses duration; decreasing speed (< 1.0)
    stretches it. Pitch shifts accordingly -- this is equivalent to changing
    the playback rate, not time-stretching.

    Args:
        audio: Input audio array (float32, mono).
        sample_rate: Native sample rate of the audio.
        speed: Speed multiplier. Must be a positive number.

    Returns:
        Resampled audio array at the original sample_rate with adjusted speed.
    """
    if speed <= 0:
        raise ValueError(f"speed must be a positive number, got {speed}")

    # Trick: pretend audio was recorded at (sr * speed) and resample back to sr.
    # This yields the same result as changing the playback rate.
    effective_sr = int(round(sample_rate * speed))
    return librosa.resample(audio, orig_sr=effective_sr, target_sr=sample_rate)
