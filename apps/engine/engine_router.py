"""
engine_router.py -- Language-based TTS engine router with VRAM swap management.

Maintains a single active engine at any time. When a request requires a
different engine than what is currently loaded, the active one is unloaded
first so that only one model occupies GPU memory at a time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from config import ENGINE_NAMES, F5_CUSTOM_CHECKPOINT, LANGUAGE_ENGINE_MAP
from engine.coqui_engine import CoquiEngine
from engine.f5_engine import F5Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine parameter capability maps
# ---------------------------------------------------------------------------

# Describes which UI / caller parameters are meaningful for each engine.
# True  = the engine accepts and uses this parameter.
# False = the parameter is irrelevant for this engine (ignored or not exposed).
_ENGINE_PARAMS: dict[str, dict[str, bool]] = {
    "f5": {
        "steps": True,
        "cfg_strength": True,
        "speed": True,
        "max_chars_chunk": True,
        "ref_text": True,
    },
    "coqui": {
        "steps": False,
        "cfg_strength": False,
        "speed": True,
        "max_chars_chunk": False,
        "ref_text": False,
    },
}


class EngineRouter:
    """Routes TTS generation requests to the correct engine by language code.

    Only one engine is kept loaded at a time. When the required engine
    differs from the currently loaded one, the active engine is unloaded
    before the new one is loaded (VRAM swap).

    Thread-safe: all state mutations are protected by a reentrant lock so
    that Gradio's threaded request handlers cannot interleave swap operations.
    """

    def __init__(self) -> None:
        # Instantiate engine objects (no model loaded yet — lazy init).
        self._engines: dict[str, F5Engine | CoquiEngine] = {
            "f5": F5Engine(custom_checkpoint=F5_CUSTOM_CHECKPOINT),
            "coqui": CoquiEngine(),
        }
        self._current_engine_key: str | None = None

        # Reentrant lock: allows the same thread to call _ensure_engine
        # recursively without dead-locking, while still blocking other threads.
        self._lock: threading.RLock = threading.RLock()

        logger.debug("EngineRouter: initialised (no model loaded yet).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        language: str,
        ref_audio: str,
        gen_text: str,
        **kwargs: Any,
    ) -> str:
        """Generate speech for the given language and return the output path.

        Routes to F5Engine for Thai ("th") and to CoquiEngine for all other
        supported languages. Performs a VRAM swap when switching engines.

        Args:
            language: BCP-47 language code (e.g. "th", "en", "ja").
            ref_audio: Absolute path to the reference audio file.
            gen_text: Text to synthesise.
            **kwargs: Engine-specific parameters forwarded verbatim.
                F5Engine accepts: ref_text, steps, cfg_strength, speed,
                    max_chars_chunk, noise_reduction.
                CoquiEngine accepts: speed, noise_reduction.
                Irrelevant keys are silently ignored by each engine.

        Returns:
            Absolute path to the generated WAV file.

        Raises:
            ValueError: If language is not in LANGUAGE_ENGINE_MAP.
            FileNotFoundError: If ref_audio does not exist.
            RuntimeError: If engine loading or inference fails.
        """
        engine_key = self._resolve_engine_key(language)

        logger.info(
            "EngineRouter.generate(): language=%r -> engine=%r | ref_audio=%r",
            language,
            engine_key,
            ref_audio,
        )

        with self._lock:
            self._ensure_engine(engine_key)
            engine = self._engines[engine_key]

        # Inject the language kwarg that CoquiEngine requires.
        # F5Engine ignores unknown kwargs, so this is safe for both paths.
        kwargs.setdefault("language", language)

        try:
            output_path = engine.generate(ref_audio, gen_text, **kwargs)
        except Exception as exc:
            logger.error(
                "EngineRouter: generation failed for language=%r engine=%r: %s",
                language,
                engine_key,
                exc,
                exc_info=True,
            )
            raise

        logger.info("EngineRouter: output saved to %s", output_path)
        return output_path

    def get_engine_name(self, language: str) -> str:
        """Return the human-readable display name for the engine used by language.

        Args:
            language: BCP-47 language code.

        Returns:
            Display name string, e.g. "F5-TTS-TH-V2" or "Coqui XTTS v2".

        Raises:
            ValueError: If language is not supported.
        """
        engine_key = self._resolve_engine_key(language)
        return ENGINE_NAMES[engine_key]

    def get_engine_params(self, language: str) -> dict[str, bool]:
        """Return a capability map indicating which parameters apply to the engine.

        Each key is a parameter name; the boolean value indicates whether
        that parameter is relevant (True) or unused (False) for the engine
        that handles the given language.

        Args:
            language: BCP-47 language code.

        Returns:
            dict with keys: steps, cfg_strength, speed, max_chars_chunk, ref_text.

        Raises:
            ValueError: If language is not supported.

        Example:
            >>> router.get_engine_params("th")
            {"steps": True, "cfg_strength": True, "speed": True,
             "max_chars_chunk": True, "ref_text": True}
            >>> router.get_engine_params("en")
            {"steps": False, "cfg_strength": False, "speed": True,
             "max_chars_chunk": False, "ref_text": False}
        """
        engine_key = self._resolve_engine_key(language)
        # Return a copy so callers cannot mutate the internal map.
        return dict(_ENGINE_PARAMS[engine_key])

    def get_current_engine_key(self) -> str | None:
        """Return the key of the currently loaded engine, or None if none is loaded.

        Returns:
            "f5", "coqui", or None.
        """
        return self._current_engine_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_engine(self, engine_key: str) -> None:
        """Load engine_key, unloading the current engine first if necessary.

        This method must be called while holding self._lock.

        Args:
            engine_key: "f5" or "coqui".

        Side effects:
            - Unloads the currently active engine when it differs from engine_key.
            - Loads the requested engine if not already loaded.
            - Updates self._current_engine_key.
        """
        if self._current_engine_key == engine_key:
            # Correct engine is already active; nothing to do.
            if not self._engines[engine_key].is_loaded:
                # Edge case: key matches but model was unloaded externally.
                logger.warning(
                    "EngineRouter: engine %r marked as current but not loaded; reloading.",
                    engine_key,
                )
                self._engines[engine_key].load_model()
            return

        # --- VRAM swap ---
        if self._current_engine_key is not None:
            current = self._engines[self._current_engine_key]
            if current.is_loaded:
                logger.info(
                    "EngineRouter: VRAM swap -- unloading %r before loading %r.",
                    self._current_engine_key,
                    engine_key,
                )
                current.unload_model()

        # --- Load the requested engine ---
        target = self._engines[engine_key]
        if not target.is_loaded:
            logger.info("EngineRouter: loading engine %r.", engine_key)
            target.load_model()

        self._current_engine_key = engine_key
        logger.info("EngineRouter: active engine is now %r.", engine_key)

    def _resolve_engine_key(self, language: str) -> str:
        """Map a language code to its engine key.

        Args:
            language: BCP-47 language code.

        Returns:
            "f5" or "coqui".

        Raises:
            ValueError: If language is not present in LANGUAGE_ENGINE_MAP.
        """
        language = language.strip().lower()
        engine_key = LANGUAGE_ENGINE_MAP.get(language)
        if engine_key is None:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported languages: {sorted(LANGUAGE_ENGINE_MAP.keys())}"
            )
        return engine_key
