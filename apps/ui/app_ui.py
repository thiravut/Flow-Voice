"""
app_ui.py — Gradio UI with 3 tabs for Multi-Language TTS Voice Cloning.

Builds the entire interface via create_ui() and wires all event handlers.
All exceptions are caught at the handler boundary so the Gradio server
never crashes.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    APP_TITLE,
    CFG_MAX,
    CFG_MIN,
    CFG_STRENGTH,
    DEFAULT_LANGUAGE,
    ENGINE_NAMES,
    HISTORY_DIR,
    INFERENCE_STEPS,
    LANGUAGE_ENGINE_MAP,
    MAX_CHARS_CHUNK,
    MAX_CHARS_MAX,
    MAX_CHARS_MIN,
    MAX_HISTORY,
    NOISE_REDUCTION,
    PRESETS_DIR,
    SPEED,
    SPEED_MAX,
    SPEED_MIN,
    STEPS_MAX,
    STEPS_MIN,
    SUPPORTED_LANGUAGES,
)
from emotions.emotion_manager import EmotionManager
from engine.audio_processor import apply_noise_reduction, validate_audio
from engine.engine_router import EngineRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered language choices for the dropdown: "th - ไทย", "en - English", …
LANGUAGE_CHOICES: list[tuple[str, str]] = [
    (f"{code} - {name}", code)
    for code, name in SUPPORTED_LANGUAGES.items()
]

_CSS_PATH = os.path.join(os.path.dirname(__file__), "custom.css")

_REF_MODE_UPLOAD = "Upload Audio"
_REF_MODE_EMOTION = "Emotion Preset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_for(language: str) -> str:
    """Return the human-readable engine name for a language code."""
    engine_key = LANGUAGE_ENGINE_MAP.get(language, "coqui")
    return ENGINE_NAMES.get(engine_key, "Coqui XTTS v2")


def _is_f5(language: str) -> bool:
    """Return True when the language uses the F5-TTS engine."""
    return LANGUAGE_ENGINE_MAP.get(language, "coqui") == "f5"


def _fmt_audio_info(info: dict) -> str:
    """Format an audio-info dict into a short human-readable string."""
    if info.get("error"):
        return f"Error: {info['error']}"
    dur = info.get("duration", 0.0)
    fmt = (info.get("format") or "?").upper()
    sr = info.get("sample_rate", 0)
    sr_khz = f"{sr / 1000:.1f}" if sr else "?"
    return f"{dur:.1f}s  |  {fmt}  |  {sr_khz} kHz"


def _load_css() -> str:
    """Read custom.css; return empty string on failure."""
    try:
        with open(_CSS_PATH, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        logger.warning("Could not load custom.css from %s", _CSS_PATH)
        return ""


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def _scan_history() -> list[list]:
    """Return up to MAX_HISTORY rows of [filename, date, size_kb] from HISTORY_DIR."""
    rows: list[list] = []
    hist_path = Path(HISTORY_DIR)
    if not hist_path.exists():
        return rows

    entries = sorted(
        (p for p in hist_path.glob("*.wav") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for entry in entries[:MAX_HISTORY]:
        try:
            stat = entry.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = round(stat.st_size / 1024, 1)
            rows.append([entry.name, mtime, f"{size_kb} KB"])
        except Exception:
            continue

    return rows


# ---------------------------------------------------------------------------
# Main UI factory
# ---------------------------------------------------------------------------


def create_ui(
    router: EngineRouter,
    emotion_manager: EmotionManager,
) -> gr.Blocks:
    """Build and return the complete Gradio Blocks UI.

    All event-handler functions are defined as closures so they can reference
    router and emotion_manager without globals.
    """
    css = _load_css()

    # ------------------------------------------------------------------
    # Internal event handlers (closures)
    # ------------------------------------------------------------------

    def on_ref_mode_change(mode: str):
        """Toggle between upload group and emotion preset group."""
        is_upload = mode == _REF_MODE_UPLOAD
        return (
            gr.update(visible=is_upload),       # upload_group
            gr.update(visible=not is_upload),   # emotion_group
        )

    def on_audio_upload(audio_path: str | None):
        """Validate uploaded/recorded audio and return info string."""
        if not audio_path:
            return ""
        try:
            info = validate_audio(audio_path)
            return _fmt_audio_info(info)
        except Exception as exc:
            logger.exception("on_audio_upload error")
            return f"Error reading audio: {exc}"

    def _build_preset_choices() -> list[tuple[str, str]]:
        """Return (display_label, preset_id) choices for emotion preset dropdown."""
        presets = emotion_manager.list_presets()
        choices: list[tuple[str, str]] = []
        for p in presets:
            label = p.get("name_en") or p.get("name_th") or p["id"]
            choices.append((label, p["id"]))
        return choices

    def on_emotion_select(preset_id: str | None):
        """Load an emotion preset: return its audio path and ref_text."""
        if not preset_id:
            return None, ""
        try:
            preset = emotion_manager.get_preset(preset_id)
            if preset is None:
                return None, ""
            audio_file = preset.get("audio_file")
            audio_path: str | None = None
            if audio_file:
                candidate = str(PRESETS_DIR / audio_file)
                if os.path.isfile(candidate):
                    audio_path = candidate
            ref_text = preset.get("ref_text", "")
            return audio_path, ref_text
        except Exception as exc:
            logger.exception("on_emotion_select error")
            return None, ""

    def generate_speech(
        language: str,
        ref_mode: str,
        upload_audio: str | None,
        emotion_preset_id: str | None,
        preset_audio_state: str | None,
        ref_text: str,
        gen_text: str,
        steps: int,
        cfg: float,
        speed: float,
        max_chars: int,
        noise_reduce: bool,
    ):
        """Core generation handler.  Returns (audio_path, status_message)."""
        # --- input validation ---
        if not gen_text or not gen_text.strip():
            return None, "Please enter text to generate."

        # Resolve reference audio path
        if ref_mode == _REF_MODE_UPLOAD:
            ref_audio = upload_audio
        else:
            ref_audio = preset_audio_state  # populated by on_emotion_select

        if not ref_audio or not os.path.isfile(str(ref_audio)):
            return None, "Please provide a valid reference audio file."

        try:
            engine_name = router.get_engine_name(language)
        except Exception:
            engine_name = _engine_for(language)

        try:
            kwargs: dict = {
                "ref_text": ref_text.strip() if ref_text else "",
                "speed": float(speed),
                "max_chars": int(max_chars),
            }
            if _is_f5(language):
                kwargs["steps"] = int(steps)
                kwargs["cfg_strength"] = float(cfg)

            output_path = router.generate(
                language=language,
                ref_audio=str(ref_audio),
                gen_text=gen_text.strip(),
                **kwargs,
            )

            if noise_reduce and output_path and os.path.isfile(output_path):
                try:
                    output_path = apply_noise_reduction(output_path, output_path)
                except Exception as exc:
                    logger.warning("Noise reduction failed: %s", exc)

            status = f"Generated using {engine_name}."
            return output_path, status

        except FileNotFoundError as exc:
            logger.error("generate_speech FileNotFoundError: %s", exc)
            return None, f"Reference audio not found: {exc}"
        except RuntimeError as exc:
            logger.error("generate_speech RuntimeError: %s", exc)
            return None, f"Generation failed: {exc}"
        except Exception as exc:
            logger.exception("generate_speech unexpected error")
            return None, f"Unexpected error: {exc}"

    def refresh_presets():
        """Reload the presets table and dropdown choices."""
        try:
            table = emotion_manager.get_presets_table()
            choices = _build_preset_choices()
            return (
                table,
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )
        except Exception as exc:
            logger.exception("refresh_presets error")
            return [], gr.update(), gr.update(), gr.update()

    def add_emotion_preset(
        add_name: str,
        add_audio: str | None,
        add_ref_text: str,
    ):
        """Add a custom emotion preset; return updated table and status."""
        if not add_name or not add_name.strip():
            return gr.update(), "Please enter a name for the preset."
        if not add_audio or not os.path.isfile(str(add_audio)):
            return gr.update(), "Please upload a valid audio file."
        try:
            emotion_manager.add_preset(
                name=add_name.strip(),
                audio_path=str(add_audio),
                ref_text=(add_ref_text or "").strip(),
            )
            table = emotion_manager.get_presets_table()
            return table, f"Preset '{add_name.strip()}' added successfully."
        except Exception as exc:
            logger.exception("add_emotion_preset error")
            return gr.update(), f"Failed to add preset: {exc}"

    def delete_emotion_preset(preset_id: str | None):
        """Delete a custom preset; return updated table and status."""
        if not preset_id:
            return gr.update(), "Please select a preset to delete."
        try:
            preset = emotion_manager.get_preset(preset_id)
            if preset is None:
                return gr.update(), "Preset not found."
            if preset.get("is_default", False):
                return gr.update(), "Cannot delete a default preset."
            success = emotion_manager.delete_preset(preset_id)
            table = emotion_manager.get_presets_table()
            if success:
                return table, f"Preset '{preset.get('name_en', preset_id)}' deleted."
            return table, "Could not delete preset."
        except Exception as exc:
            logger.exception("delete_emotion_preset error")
            return gr.update(), f"Failed to delete preset: {exc}"

    def load_history():
        """Scan HISTORY_DIR and return fresh table rows."""
        try:
            return _scan_history()
        except Exception as exc:
            logger.exception("load_history error")
            return []

    def play_history_item(evt: gr.SelectData, history_data):
        """Return the audio path for the selected history row."""
        try:
            if evt.index is None:
                return None
            row_idx = evt.index[0]
            rows = history_data if isinstance(history_data, list) else []
            if row_idx >= len(rows):
                return None
            filename = rows[row_idx][0]
            audio_path = os.path.join(HISTORY_DIR, filename)
            if os.path.isfile(audio_path):
                return audio_path
            return None
        except Exception as exc:
            logger.exception("play_history_item error")
            return None

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    with gr.Blocks(title=APP_TITLE, css=css) as demo:

        # ---- app title ----
        gr.Markdown(f"# {APP_TITLE}")

        with gr.Tabs():

            # ==============================================================
            # TAB 1 — Generate Speech
            # ==============================================================
            with gr.Tab("Generate Speech"):

                # ---- language / engine row ----
                with gr.Row(elem_id="lang-engine-row"):
                    language_dd = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGE_CHOICES,
                        value=DEFAULT_LANGUAGE,
                        interactive=True,
                        scale=2,
                    )
                    engine_label = gr.Textbox(
                        label="Engine",
                        value=_engine_for(DEFAULT_LANGUAGE),
                        interactive=False,
                        elem_id="engine-display",
                        scale=1,
                    )

                # ---- two-column body ----
                with gr.Row(elem_id="generate-columns"):

                    # ---- LEFT: Reference Voice ----
                    with gr.Column(elem_id="ref-voice-panel"):
                        gr.Markdown("### Reference Voice")

                        ref_mode = gr.Radio(
                            choices=[_REF_MODE_UPLOAD, _REF_MODE_EMOTION],
                            value=_REF_MODE_UPLOAD,
                            label="Mode",
                            elem_id="ref-mode-radio",
                        )

                        # Upload sub-group
                        with gr.Group(visible=True) as upload_group:
                            ref_audio_upload = gr.Audio(
                                label="Reference Audio",
                                sources=["upload", "microphone"],
                                type="filepath",
                                elem_id="ref-audio-upload",
                            )

                        # Emotion preset sub-group
                        with gr.Group(visible=False) as emotion_group:
                            emotion_dd = gr.Dropdown(
                                label="Emotion Preset",
                                choices=_build_preset_choices(),
                                value=None,
                                interactive=True,
                            )
                            # Hidden audio player that shows preset preview
                            emotion_preview_audio = gr.Audio(
                                label="Preset Preview",
                                interactive=False,
                                visible=True,
                            )

                        # Audio info strip
                        audio_info = gr.Textbox(
                            label="Audio Info",
                            value="",
                            interactive=False,
                            elem_id="audio-info-display",
                            max_lines=1,
                        )

                        # ref_text — shown only for F5 / Thai
                        with gr.Row(
                            visible=_is_f5(DEFAULT_LANGUAGE)
                        ) as ref_text_row:
                            ref_text_input = gr.Textbox(
                                label="Reference Text",
                                placeholder="Transcript of the reference audio (optional but recommended)",
                                lines=2,
                                elem_id="ref-text-input",
                            )

                    # ---- RIGHT: Generation Settings ----
                    with gr.Column(elem_id="gen-settings-panel"):
                        gr.Markdown("### Generation Settings")

                        # F5-specific parameters group
                        with gr.Group(
                            visible=_is_f5(DEFAULT_LANGUAGE),
                            elem_id="f5-params-group",
                        ) as f5_params_group:
                            steps_slider = gr.Slider(
                                label="Inference Steps",
                                minimum=STEPS_MIN,
                                maximum=STEPS_MAX,
                                value=INFERENCE_STEPS,
                                step=1,
                            )
                            cfg_slider = gr.Slider(
                                label="CFG Strength",
                                minimum=CFG_MIN,
                                maximum=CFG_MAX,
                                value=CFG_STRENGTH,
                                step=0.1,
                            )

                        # Common parameters (always visible)
                        speed_slider = gr.Slider(
                            label="Speed",
                            minimum=SPEED_MIN,
                            maximum=SPEED_MAX,
                            value=SPEED,
                            step=0.05,
                        )

                        # Max chars — F5 / chunking relevant for all engines
                        with gr.Group(
                            visible=_is_f5(DEFAULT_LANGUAGE)
                        ) as max_chars_group:
                            max_chars_slider = gr.Slider(
                                label="Max Characters per Chunk",
                                minimum=MAX_CHARS_MIN,
                                maximum=MAX_CHARS_MAX,
                                value=MAX_CHARS_CHUNK,
                                step=10,
                            )

                        noise_reduce_check = gr.Checkbox(
                            label="Noise Reduction",
                            value=NOISE_REDUCTION,
                            elem_id="noise-reduce-check",
                        )

                # ---- text input ----
                gen_text_input = gr.Textbox(
                    label="Text to Generate",
                    placeholder="Enter the text you want to synthesize...",
                    lines=3,
                    elem_id="text-input",
                )

                # ---- generate button ----
                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    elem_id="generate-btn",
                )

                # ---- output ----
                output_audio = gr.Audio(
                    label="Output Audio",
                    interactive=False,
                    elem_id="output-audio",
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    elem_id="status-text",
                    max_lines=2,
                )

                # Hidden state: stores audio path populated by emotion selection
                preset_audio_state = gr.State(value=None)

                # ------------------------------------------------------------------
                # Tab 1 event wiring
                # ------------------------------------------------------------------

                def on_language_change_full(language: str):
                    """Update all language-dependent UI elements in one handler."""
                    engine_name = _engine_for(language)
                    f5 = _is_f5(language)
                    return (
                        engine_name,
                        gr.update(visible=f5),   # ref_text_row
                        gr.update(visible=f5),   # f5_params_group
                        gr.update(visible=f5),   # max_chars_group
                    )

                language_dd.change(
                    fn=on_language_change_full,
                    inputs=[language_dd],
                    outputs=[engine_label, ref_text_row, f5_params_group, max_chars_group],
                )

                ref_mode.change(
                    fn=on_ref_mode_change,
                    inputs=[ref_mode],
                    outputs=[upload_group, emotion_group],
                )

                ref_audio_upload.change(
                    fn=on_audio_upload,
                    inputs=[ref_audio_upload],
                    outputs=[audio_info],
                )

                def on_emotion_select_ui(preset_id: str | None):
                    audio_path, ref_text = on_emotion_select(preset_id)
                    info_str = ""
                    if audio_path:
                        try:
                            from engine.audio_processor import get_audio_info
                            info = get_audio_info(audio_path)
                            info_str = _fmt_audio_info(info)
                        except Exception:
                            pass
                    return audio_path, ref_text, audio_path, info_str

                emotion_dd.change(
                    fn=on_emotion_select_ui,
                    inputs=[emotion_dd],
                    outputs=[
                        emotion_preview_audio,
                        ref_text_input,
                        preset_audio_state,
                        audio_info,
                    ],
                )

                generate_btn.click(
                    fn=generate_speech,
                    inputs=[
                        language_dd,
                        ref_mode,
                        ref_audio_upload,
                        emotion_dd,
                        preset_audio_state,
                        ref_text_input,
                        gen_text_input,
                        steps_slider,
                        cfg_slider,
                        speed_slider,
                        max_chars_slider,
                        noise_reduce_check,
                    ],
                    outputs=[output_audio, status_text],
                )

            # ==============================================================
            # TAB 2 — Emotion Presets
            # ==============================================================
            with gr.Tab("Emotion Presets"):

                gr.Markdown("## Emotion Presets")
                gr.Markdown(
                    "Browse and manage emotion presets used as reference voices."
                )

                with gr.Row():
                    refresh_presets_btn = gr.Button("Refresh", variant="secondary")

                presets_table = gr.DataFrame(
                    value=emotion_manager.get_presets_table(),
                    headers=["Name (EN)", "Name (TH)", "Description", "Has Audio", "Ref Text"],
                    datatype=["str", "str", "str", "bool", "str"],
                    interactive=False,
                    label="All Presets",
                    elem_id="emotion-table",
                )

                gr.Markdown("### Preview Preset")
                with gr.Row():
                    preview_dd = gr.Dropdown(
                        label="Select Preset to Preview",
                        choices=_build_preset_choices(),
                        value=None,
                        interactive=True,
                        scale=3,
                    )
                    preview_btn = gr.Button("Load Preview", variant="secondary", scale=1)

                preview_audio = gr.Audio(
                    label="Preset Audio Preview",
                    interactive=False,
                )
                preview_ref_text = gr.Textbox(
                    label="Preset Reference Text",
                    interactive=False,
                    lines=2,
                )

                # ---- Add custom preset ----
                with gr.Group(elem_id="add-emotion-group"):
                    gr.Markdown("### Add Custom Preset")
                    with gr.Row():
                        add_name_input = gr.Textbox(
                            label="Preset Name",
                            placeholder="e.g. Calm, Excited, Narrator",
                            scale=2,
                        )
                    with gr.Row():
                        add_audio_input = gr.Audio(
                            label="Reference Audio",
                            sources=["upload", "microphone"],
                            type="filepath",
                            scale=3,
                        )
                    add_ref_text_input = gr.Textbox(
                        label="Reference Text (optional)",
                        placeholder="Transcript of the reference audio",
                        lines=2,
                    )
                    add_preset_btn = gr.Button("Add Preset", variant="primary")
                    add_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=2,
                    )

                # ---- Delete custom preset ----
                with gr.Group():
                    gr.Markdown("### Delete Custom Preset")
                    with gr.Row():
                        delete_dd = gr.Dropdown(
                            label="Select Preset to Delete",
                            choices=_build_preset_choices(),
                            value=None,
                            interactive=True,
                            scale=3,
                        )
                        delete_btn = gr.Button("Delete", variant="stop", scale=1)
                    delete_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=2,
                    )

                # ------------------------------------------------------------------
                # Tab 2 event wiring
                # ------------------------------------------------------------------

                def handle_refresh_presets():
                    try:
                        table = emotion_manager.get_presets_table()
                        choices = _build_preset_choices()
                        return (
                            table,
                            gr.update(choices=choices, value=None),
                            gr.update(choices=choices, value=None),
                            gr.update(choices=choices, value=None),
                        )
                    except Exception as exc:
                        logger.exception("handle_refresh_presets error")
                        return gr.update(), gr.update(), gr.update(), gr.update()

                refresh_presets_btn.click(
                    fn=handle_refresh_presets,
                    inputs=[],
                    outputs=[presets_table, preview_dd, delete_dd, emotion_dd],
                )

                def handle_preview(preset_id: str | None):
                    audio_path, ref_text = on_emotion_select(preset_id)
                    return audio_path, ref_text

                preview_btn.click(
                    fn=handle_preview,
                    inputs=[preview_dd],
                    outputs=[preview_audio, preview_ref_text],
                )

                preview_dd.change(
                    fn=handle_preview,
                    inputs=[preview_dd],
                    outputs=[preview_audio, preview_ref_text],
                )

                add_preset_btn.click(
                    fn=add_emotion_preset,
                    inputs=[add_name_input, add_audio_input, add_ref_text_input],
                    outputs=[presets_table, add_status],
                )

                def handle_delete(preset_id: str | None):
                    table, msg = delete_emotion_preset(preset_id)
                    choices = _build_preset_choices()
                    return (
                        table,
                        msg,
                        gr.update(choices=choices, value=None),
                        gr.update(choices=choices, value=None),
                        gr.update(choices=choices, value=None),
                    )

                delete_btn.click(
                    fn=handle_delete,
                    inputs=[delete_dd],
                    outputs=[
                        presets_table,
                        delete_status,
                        delete_dd,
                        preview_dd,
                        emotion_dd,
                    ],
                )

            # ==============================================================
            # TAB 3 — History
            # ==============================================================
            with gr.Tab("History"):

                gr.Markdown("## Generation History")
                gr.Markdown(
                    "Click any row to load the audio. "
                    f"Showing the {MAX_HISTORY} most recent files."
                )

                with gr.Row():
                    refresh_history_btn = gr.Button("Refresh", variant="secondary")

                history_table = gr.DataFrame(
                    value=_scan_history(),
                    headers=["Filename", "Date", "Size"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    label="History",
                    elem_id="history-table",
                )

                history_audio = gr.Audio(
                    label="Selected Audio",
                    interactive=False,
                )

                # ------------------------------------------------------------------
                # Tab 3 event wiring
                # ------------------------------------------------------------------

                refresh_history_btn.click(
                    fn=load_history,
                    inputs=[],
                    outputs=[history_table],
                )

                history_table.select(
                    fn=play_history_item,
                    inputs=[history_table],
                    outputs=[history_audio],
                )

    return demo
