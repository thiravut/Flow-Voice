"""
emotion_manager.py — CRUD operations for emotion presets.

Manages the lifecycle of emotion presets: loading, listing, adding custom
presets (with audio file copy), deleting custom presets, and persisting
state back to metadata.json.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

from config import EMOTION_METADATA_FILE, PRESETS_DIR


class EmotionManager:
    """Manages emotion presets backed by a metadata.json file."""

    def __init__(self) -> None:
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        self._presets: list[dict] = []
        self._load_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_presets(self) -> list[dict]:
        """Return all presets (default + custom) as a list of dicts."""
        return list(self._presets)

    def get_preset(self, preset_id: str) -> dict | None:
        """Return a single preset by ID, or None if not found."""
        for preset in self._presets:
            if preset["id"] == preset_id:
                return dict(preset)
        return None

    def add_preset(
        self,
        name: str,
        audio_path: str,
        ref_text: str = "",
    ) -> dict:
        """Create a new custom preset and persist it.

        Copies the audio file into PRESETS_DIR as custom_{name}_{hash8}.wav,
        generates a unique ID from the name, appends the preset to the list,
        and saves metadata.json.
        """
        clean_name = self._slugify(name)
        preset_id = self._unique_id(clean_name)

        src = Path(audio_path)
        audio_hash = self._file_hash8(src)
        dest_filename = f"custom_{clean_name}_{audio_hash}.wav"
        dest_path = PRESETS_DIR / dest_filename
        shutil.copy2(src, dest_path)

        new_preset: dict = {
            "id": preset_id,
            "name_en": name,
            "name_th": name,
            "description": "",
            "ref_text": ref_text,
            "audio_file": dest_filename,
            "is_default": False,
        }

        self._presets.append(new_preset)
        self._save_metadata()
        return dict(new_preset)

    def delete_preset(self, preset_id: str) -> bool:
        """Delete a custom preset by ID. Returns True on success.

        Default presets (is_default=True) cannot be deleted.
        The associated audio file is removed if it exists.
        """
        for i, preset in enumerate(self._presets):
            if preset["id"] == preset_id:
                if preset.get("is_default", False):
                    return False

                audio_file = preset.get("audio_file")
                if audio_file:
                    audio_path = PRESETS_DIR / audio_file
                    if audio_path.exists():
                        audio_path.unlink()

                self._presets.pop(i)
                self._save_metadata()
                return True

        return False

    def get_presets_table(self) -> list[list]:
        """Return preset data as rows for a Gradio DataFrame.

        Columns: Name (EN), Name (TH), Description, Has Audio, Ref Text
        """
        rows: list[list] = []
        for preset in self._presets:
            audio_file = preset.get("audio_file")
            has_audio = bool(
                audio_file and (PRESETS_DIR / audio_file).exists()
            )
            rows.append(
                [
                    preset.get("name_en", ""),
                    preset.get("name_th", ""),
                    preset.get("description", ""),
                    has_audio,
                    preset.get("ref_text", ""),
                ]
            )
        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Load presets from metadata.json into memory."""
        if not EMOTION_METADATA_FILE.exists():
            self._presets = []
            return

        with EMOTION_METADATA_FILE.open(encoding="utf-8") as fh:
            data: dict = json.load(fh)

        self._presets = data.get("presets", [])

    def _save_metadata(self) -> None:
        """Write current in-memory presets back to metadata.json."""
        data = {"presets": self._presets}
        with EMOTION_METADATA_FILE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a display name to a lowercase ASCII slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        slug = slug.strip("_")
        return slug or "preset"

    def _unique_id(self, base: str) -> str:
        """Return a preset ID derived from base that does not collide."""
        existing_ids = {p["id"] for p in self._presets}
        candidate = base
        counter = 2
        while candidate in existing_ids:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    @staticmethod
    def _file_hash8(path: Path) -> str:
        """Return the first 8 hex characters of the SHA-256 of a file."""
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:8]
