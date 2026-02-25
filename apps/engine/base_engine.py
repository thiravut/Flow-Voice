"""
base_engine.py — Abstract Base Class defining the interface all TTS engines must implement.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import numpy as np
import soundfile as sf

from config import HISTORY_DIR, OUTPUT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engine implementations.

    Subclasses must implement load_model, unload_model, generate,
    and get_supported_languages.
    """

    # Class-level lock shared across all engine instances to prevent
    # concurrent CUDA inference and potential OOM errors.
    _generate_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded into GPU memory."""
        return self._loaded

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model onto the GPU."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free VRAM."""
        ...

    @abstractmethod
    def generate(self, ref_audio: str, gen_text: str, **kwargs) -> str:
        """Generate speech and return the saved output file path.

        Args:
            ref_audio: Path to reference audio file.
            gen_text: Text to synthesize.
            **kwargs: Engine-specific parameters (steps, speed, cfg, etc.).

        Returns:
            Absolute path to the generated WAV file saved in HISTORY_DIR.
        """
        ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of BCP-47 language codes supported by this engine."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _save_to_history(self, audio_data: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> str:
        """Save audio array to HISTORY_DIR as a WAV file and return its path.

        Filename format: YYYYMMDD_HHMMSS_<hash8>.wav
        where hash8 is the first 8 characters of the MD5 of the raw audio bytes.
        """
        os.makedirs(HISTORY_DIR, exist_ok=True)

        raw_bytes = audio_data.tobytes()
        hash8 = hashlib.md5(raw_bytes).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{hash8}.wav"
        output_path = os.path.join(HISTORY_DIR, filename)

        sf.write(output_path, audio_data, sample_rate)
        logger.info("Saved generated audio to %s", output_path)
        return output_path

    @contextmanager
    def _with_lock(self) -> Generator[None, None, None]:
        """Context manager that acquires the class-level generation lock."""
        logger.debug("%s waiting for generate lock", self.__class__.__name__)
        with self.__class__._generate_lock:
            logger.debug("%s acquired generate lock", self.__class__.__name__)
            yield
        logger.debug("%s released generate lock", self.__class__.__name__)

    def _safe_generate(self, ref_audio: str, gen_text: str, **kwargs) -> str:
        """Wraps generate() with lock acquisition and unified exception handling.

        Intended as an optional helper for subclasses that prefer a single
        protected call site rather than managing the lock themselves.
        """
        with self._with_lock():
            try:
                return self.generate(ref_audio, gen_text, **kwargs)
            except FileNotFoundError as exc:
                logger.error("Reference audio not found: %s", exc)
                raise
            except RuntimeError as exc:
                logger.error("Runtime error during generation: %s", exc)
                raise
            except Exception as exc:
                logger.error(
                    "Unexpected error in %s.generate(): %s",
                    self.__class__.__name__,
                    exc,
                    exc_info=True,
                )
                raise
