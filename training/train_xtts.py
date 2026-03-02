"""
train_xtts.py

Fine-tune Coqui XTTS v2 on a prepared dataset.

Follows the training pattern from TTS.demos.xtts_ft_demo.utils.gpt_train.

Usage:
    python training/train_xtts.py --config training/config/xtts_finetune.yaml
    python training/train_xtts.py --config training/config/xtts_finetune.yaml --epochs 1

WARNING: Thai (th) is NOT natively supported by the XTTS v2 BPE tokenizer.
VoiceBpeTokenizer.preprocess_text() raises NotImplementedError for lang="th".
Options:
  1. Patch the tokenizer (recommended): add Thai character handling in the
     preprocess_text() method of VoiceBpeTokenizer in the installed Coqui TTS
     package. Thai uses basic_cleaners (no special preprocessing needed), so
     the simplest patch is to add "th" to the languages that fall through to
     the basic_cleaners branch, similar to how "hi" is handled.
  2. Pre-romanize Thai text before running prepare_xtts_dataset.py using
     pythainlp.transliterate and set language: "en" in the YAML config.
     Quality will be lower.
  3. Use F5-TTS-TH-V2 instead, which natively supports Thai.

This script will attempt training with the configured language and raise a
clear error if the tokenizer rejects it.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import urllib.request
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pretrained checkpoint download URLs
# ---------------------------------------------------------------------------

DVAE_CHECKPOINT_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
)
MEL_NORM_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
)
TOKENIZER_FILE_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
)
XTTS_CHECKPOINT_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
)
XTTS_CONFIG_URL = (
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
)

# Local TTS cache installed by Coqui TTS (populated after first tts download)
def _get_local_tts_cache() -> str:
    """Return the platform-appropriate TTS model cache directory."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(home, "AppData", "Local", "tts",
                            "tts_models--multilingual--multi-dataset--xtts_v2")
    # Linux / macOS (RunPod uses Linux)
    return os.path.join(home, ".local", "share", "tts",
                        "tts_models--multilingual--multi-dataset--xtts_v2")

_LOCAL_TTS_CACHE = _get_local_tts_cache()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_yaml_config(config_path: str) -> dict:
    """Load and return the YAML training config."""
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")
    return cfg


def resolve_path(base_dir: str, path: str) -> str:
    """Resolve a path that may be relative to base_dir."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


# ---------------------------------------------------------------------------
# Checkpoint / download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: str) -> None:
    """Download url to dest, showing a simple progress indicator."""
    log.info("Downloading %s -> %s", url, dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r  {pct:5.1f}%  {downloaded // 1024 // 1024} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress


def _ensure_file(dest: str, url: str, description: str) -> str:
    """Return dest if it already exists; otherwise download it."""
    if os.path.isfile(dest):
        log.info("Found %s at %s", description, dest)
        return dest
    log.info("%s not found locally. Downloading ...", description)
    _download_file(url, dest)
    return dest


def locate_or_download_checkpoints(checkpoints_dir: str) -> dict[str, str]:
    """Ensure all required XTTS v2 checkpoint files exist.

    Checks the local TTS cache first, then falls back to downloading.
    Returns a dict with keys: dvae, mel_norm, tokenizer, xtts, config.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Required files and their download URLs
    files = {
        "dvae": ("dvae.pth", DVAE_CHECKPOINT_URL),
        "mel_norm": ("mel_stats.pth", MEL_NORM_URL),
        "tokenizer": ("vocab.json", TOKENIZER_FILE_URL),
        "xtts": ("model.pth", XTTS_CHECKPOINT_URL),
        "config": ("config.json", XTTS_CONFIG_URL),
    }

    result: dict[str, str] = {}

    for key, (filename, url) in files.items():
        # Priority 1: already present in checkpoints_dir
        target = os.path.join(checkpoints_dir, filename)
        if os.path.isfile(target):
            log.info("Found %s in checkpoints dir.", filename)
            result[key] = target
            continue

        # Priority 2: present in the local TTS model cache
        cached = os.path.join(_LOCAL_TTS_CACHE, filename)
        if os.path.isfile(cached):
            log.info("Found %s in local TTS cache: %s", filename, cached)
            result[key] = cached
            continue

        # Priority 3: download
        log.info("%s not found. Downloading from Coqui CDN ...", filename)
        _download_file(url, target)
        result[key] = target

    return result


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_dataset(dataset_path: str) -> tuple[str, str]:
    """Validate dataset directory and return (train_csv, eval_csv) paths."""
    train_csv = os.path.join(dataset_path, "metadata_train.csv")
    eval_csv = os.path.join(dataset_path, "metadata_eval.csv")
    wavs_dir = os.path.join(dataset_path, "wavs")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}\n"
            "Run prepare_xtts_dataset.py first."
        )
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(
            f"Training CSV not found: {train_csv}\n"
            "Run prepare_xtts_dataset.py first."
        )
    if not os.path.isfile(eval_csv):
        raise FileNotFoundError(
            f"Eval CSV not found: {eval_csv}\n"
            "Run prepare_xtts_dataset.py first."
        )
    if not os.path.isdir(wavs_dir):
        raise FileNotFoundError(
            f"WAV directory not found: {wavs_dir}\n"
            "Run prepare_xtts_dataset.py first."
        )

    # Count samples for a sanity check
    with open(train_csv, encoding="utf-8") as fh:
        train_count = sum(1 for _ in fh) - 1  # subtract header
    with open(eval_csv, encoding="utf-8") as fh:
        eval_count = sum(1 for _ in fh) - 1

    log.info("Dataset: %d train samples, %d eval samples", train_count, eval_count)

    if train_count < 1:
        raise ValueError("Training CSV contains no samples.")
    if eval_count < 1:
        raise ValueError(
            "Eval CSV contains no samples. "
            "Increase dataset size or lower eval_split."
        )

    return train_csv, eval_csv


# ---------------------------------------------------------------------------
# Thai tokenizer patch
# ---------------------------------------------------------------------------


def _patch_tokenizer_for_thai() -> None:
    """Attempt to monkey-patch VoiceBpeTokenizer to support lang='th'.

    The XTTS v2 tokenizer raises NotImplementedError for Thai because Thai
    is not in its language dispatch table. Thai text needs no special
    preprocessing (basic_cleaners is sufficient), so we add 'th' to the
    set of languages that pass through unchanged.
    """
    try:
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer  # type: ignore

        original_preprocess = VoiceBpeTokenizer.preprocess_text

        def _patched_preprocess(self, txt: str, lang: str) -> str:  # type: ignore
            if lang == "th":
                # Thai needs no romanization or special normalization.
                # Return text as-is; the BPE tokenizer handles byte-level
                # encoding so Unicode Thai characters are tokenized correctly.
                return txt
            return original_preprocess(self, txt, lang)

        VoiceBpeTokenizer.preprocess_text = _patched_preprocess  # type: ignore
        log.info(
            "Applied Thai tokenizer patch: lang='th' will use basic pass-through."
        )
    except ImportError:
        log.warning(
            "Could not import VoiceBpeTokenizer to apply Thai patch. "
            "Training may fail if the tokenizer rejects lang='th'."
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Thai tokenizer patch failed: %s", exc)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_training(cfg: dict, epochs_override: int | None = None) -> None:
    """Build GPTTrainer config and launch Coqui Trainer.fit()."""
    # ------------------------------------------------------------------
    # Import heavy dependencies here so --help is fast
    # ------------------------------------------------------------------
    try:
        from trainer import Trainer, TrainerArgs  # type: ignore
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
        from TTS.tts.datasets import load_tts_samples  # type: ignore
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (  # type: ignore
            GPTArgs,
            GPTTrainer,
            GPTTrainerConfig,
            XttsAudioConfig,
        )
    except ImportError as exc:
        log.error(
            "Required training dependency not found: %s\n"
            "Install training dependencies with:\n"
            "  pip install -r training/requirements_train.txt",
            exc,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve config values
    # ------------------------------------------------------------------
    # Resolve relative paths from the script's directory (training/), not the
    # config file directory (training/config/), so "../training_data" resolves
    # to the project-root level training_data/ folder.
    config_dir = os.path.dirname(os.path.abspath(__file__))

    raw_dataset_path = cfg.get("dataset_path", "../training_data/xtts")
    dataset_path = resolve_path(config_dir, raw_dataset_path)

    raw_output_dir = cfg.get("output_dir", "../checkpoints/xtts")
    output_dir = resolve_path(config_dir, raw_output_dir)

    language: str = cfg.get("language", "th")
    num_epochs: int = epochs_override if epochs_override is not None else int(cfg.get("epochs", 50))
    batch_size: int = int(cfg.get("batch_size", 2))
    learning_rate: float = float(cfg.get("learning_rate", 5e-6))
    grad_accum_steps: int = int(cfg.get("grad_accumulation_steps", 4))

    log.info("--- Training Configuration ---")
    log.info("  dataset_path : %s", dataset_path)
    log.info("  output_dir   : %s", output_dir)
    log.info("  language     : %s", language)
    log.info("  epochs       : %d", num_epochs)
    log.info("  batch_size   : %d", batch_size)
    log.info("  learning_rate: %g", learning_rate)
    log.info("  grad_accum   : %d", grad_accum_steps)
    log.info("------------------------------")

    # ------------------------------------------------------------------
    # Thai language warning
    # ------------------------------------------------------------------
    if language == "th":
        log.warning(
            "THAI LANGUAGE WARNING: Thai (th) is NOT natively supported by the "
            "XTTS v2 BPE tokenizer. A monkey-patch will be applied to allow "
            "pass-through tokenization. If training fails with a tokenizer error, "
            "see the module docstring for alternative approaches."
        )
        _patch_tokenizer_for_thai()

    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------
    train_csv, eval_csv = validate_dataset(dataset_path)

    # ------------------------------------------------------------------
    # Locate / download pretrained checkpoints
    # ------------------------------------------------------------------
    # Store downloaded extras alongside the output checkpoints
    extras_dir = os.path.join(output_dir, "pretrained")
    checkpoint_paths = locate_or_download_checkpoints(extras_dir)

    dvae_checkpoint = checkpoint_paths["dvae"]
    mel_norm_file = checkpoint_paths["mel_norm"]
    tokenizer_file = checkpoint_paths["tokenizer"]
    xtts_checkpoint = checkpoint_paths["xtts"]

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model args
    # ------------------------------------------------------------------
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 s at 22050 Hz
        min_conditioning_length=66150,    # 3 s at 22050 Hz
        debug_loading_failures=False,
        max_wav_length=255995,            # ~11.6 s at 22050 Hz
        max_text_length=200,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # ------------------------------------------------------------------
    # Build audio config
    # ------------------------------------------------------------------
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # ------------------------------------------------------------------
    # Build trainer config
    # ------------------------------------------------------------------
    # lr_scheduler milestones are scaled by the grad_accum factor so that
    # the effective step count matches the original recipe.
    scale = grad_accum_steps
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=output_dir,
        model_args=model_args,
        run_name="GPT_XTTS_FT",
        project_name="XTTS_trainer",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2,
        },
        lr=learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * scale, 150000 * scale, 300000 * scale],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[],
    )

    # ------------------------------------------------------------------
    # Dataset config
    # ------------------------------------------------------------------
    # The "coqui" formatter expects pipe-delimited CSV with columns
    # audio_file|text|speaker_name as produced by prepare_xtts_dataset.py.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=dataset_path,
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # ------------------------------------------------------------------
    # Initialize model and load samples
    # ------------------------------------------------------------------

    # Patch torch.load for Coqui TTS compatibility with PyTorch >= 2.6
    # Coqui's load_fsspec() calls torch.load without weights_only=False,
    # but XTTS checkpoints contain pickled config objects that require it.
    # We must patch torch.load directly because TTS modules already hold
    # a reference to the original load_fsspec at import time.
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    log.info("Initializing GPTTrainer from config ...")
    model = GPTTrainer.init_from_config(config)

    log.info("Loading TTS samples ...")
    try:
        train_samples, eval_samples = load_tts_samples(
            [config_dataset],
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
    except NotImplementedError as exc:
        log.error(
            "Tokenizer raised NotImplementedError for language '%s': %s\n"
            "The Thai tokenizer patch did not take effect or another language "
            "is unsupported.\n"
            "See the module docstring for workarounds.",
            language,
            exc,
        )
        sys.exit(1)

    log.info(
        "Loaded %d train samples and %d eval samples.",
        len(train_samples),
        len(eval_samples),
    )

    # Free memory before training starts
    gc.collect()

    # ------------------------------------------------------------------
    # Build and run Trainer
    # ------------------------------------------------------------------
    log.info("Starting Trainer.fit() ...")
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_accum_steps,
        ),
        config,
        output_path=output_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    log.info("Training complete. Checkpoints saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Coqui XTTS v2 on a prepared dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/train_xtts.py "
            "--config training/config/xtts_finetune.yaml\n"
            "  python training/train_xtts.py "
            "--config training/config/xtts_finetune.yaml --epochs 1\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML training config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config)
    # Stash the resolved config path so run_training() can compute relative paths
    cfg["_config_path"] = os.path.abspath(args.config)

    run_training(cfg, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
