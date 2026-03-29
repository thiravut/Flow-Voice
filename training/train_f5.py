"""
F5-TTS-TH-V2 Fine-tuning Script

Usage:
    # Single GPU
    python training/train_f5.py --config training/config/f5_finetune.yaml

    # With accelerate (recommended)
    accelerate launch training/train_f5.py --config training/config/f5_finetune.yaml

    # Short test run
    python training/train_f5.py --config training/config/f5_finetune.yaml --max-steps 10

    # Save disk space (no mel cache, ~150GB+ smaller)
    python training/train_f5.py --config training/config/f5_finetune.yaml --no-mel-cache

    # Delete existing mel cache to free disk, then train without it
    python training/train_f5.py --config training/config/f5_finetune.yaml --delete-mel-cache --no-mel-cache
"""

import argparse
import csv
import glob
import logging
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "pretrained_checkpoint": None,
    "use_ema": True,
    "dataset_path": "../training_data/f5",
    "vocab_file": "../training_data/f5/vocab.txt",
    "learning_rate": 1.0e-5,
    "batch_size_per_gpu": 1600,
    "max_samples": 8,
    "epochs": 100,
    "num_warmup_updates": 100,
    "grad_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",
    "save_per_updates": 500,
    "output_dir": "../checkpoints/f5",
    "wandb_project": "f5-tts-thai-finetune",
    "wandb_mode": "offline",
}

V2_MODEL_CFG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    text_mask_padding=True,
    conv_layers=4,
    pe_attn_head=None,
)

MEL_SPEC_KWARGS = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    mel_spec_type="vocos",
)

ODEINT_KWARGS = dict(method="euler")

HF_REPO_ID = "VIZINTZOR/F5-TTS-TH-V2"
HF_MODEL_FILENAME = "model_350000.pt"
HF_VOCAB_FILENAME = "vocab.txt"


# ---------------------------------------------------------------------------
# Pretrained path detection
# ---------------------------------------------------------------------------

def _find_hf_cache_path() -> tuple[str | None, str | None]:
    """Search HuggingFace hub cache for the V2 checkpoint and vocab files."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    # Repo folders are named models--<owner>--<repo>
    repo_folder_name = "models--VIZINTZOR--F5-TTS-TH-V2"
    repo_dir = os.path.join(hf_cache, repo_folder_name)

    if not os.path.isdir(repo_dir):
        return None, None

    # Walk snapshots to find the model file
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None, None

    ckpt_path = None
    vocab_path = None

    for snapshot in sorted(os.listdir(snapshots_dir), reverse=True):
        snapshot_dir = os.path.join(snapshots_dir, snapshot)
        candidate_ckpt = os.path.join(snapshot_dir, HF_MODEL_FILENAME)
        candidate_vocab = os.path.join(snapshot_dir, HF_VOCAB_FILENAME)
        if os.path.isfile(candidate_ckpt):
            ckpt_path = candidate_ckpt
        if os.path.isfile(candidate_vocab):
            vocab_path = candidate_vocab
        if ckpt_path and vocab_path:
            break

    return ckpt_path, vocab_path


def resolve_pretrained_checkpoint(cfg: dict) -> tuple[str, str]:
    """Return (ckpt_path, vocab_path), downloading from HF if needed."""
    ckpt_path: str | None = cfg.get("pretrained_checkpoint")
    vocab_file: str | None = cfg.get("vocab_file")

    # --- Checkpoint ---
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        logger.info("Pretrained checkpoint not specified or not found, searching HuggingFace cache...")
        cached_ckpt, cached_vocab = _find_hf_cache_path()

        if cached_ckpt:
            logger.info(f"Found cached checkpoint: {cached_ckpt}")
            ckpt_path = cached_ckpt
        else:
            logger.info(f"Downloading checkpoint from HuggingFace: {HF_REPO_ID}/{HF_MODEL_FILENAME}")
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
                logger.info(f"Downloaded to: {ckpt_path}")
                if cached_vocab is None:
                    cached_vocab = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_VOCAB_FILENAME)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not locate or download pretrained checkpoint. "
                    f"Please set pretrained_checkpoint in your config. Error: {exc}"
                ) from exc

        if cached_vocab and (vocab_file is None or not os.path.isfile(vocab_file)):
            vocab_file = cached_vocab
    else:
        # If ckpt given but vocab not given, still try cache
        if vocab_file is None or not os.path.isfile(vocab_file):
            _, cached_vocab = _find_hf_cache_path()
            if cached_vocab:
                vocab_file = cached_vocab

    # --- Vocab ---
    if vocab_file is None or not os.path.isfile(vocab_file):
        logger.info(f"Vocab file not found, downloading from HuggingFace: {HF_REPO_ID}/{HF_VOCAB_FILENAME}")
        try:
            from huggingface_hub import hf_hub_download
            vocab_file = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_VOCAB_FILENAME)
            logger.info(f"Downloaded vocab to: {vocab_file}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not locate or download vocab file. "
                f"Please set vocab_file in your config. Error: {exc}"
            ) from exc

    return ckpt_path, vocab_file


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(vocab_file: str) -> torch.nn.Module:
    """Build CFM model from V2 config and vocab."""
    from f5_tts_th import CFM, DiT
    from f5_tts_th.utils import get_tokenizer

    logger.info(f"Building model with vocab: {vocab_file}")
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    logger.info(f"Vocab size: {vocab_size}")

    model = CFM(
        transformer=DiT(
            **V2_MODEL_CFG,
            text_num_embeds=vocab_size,
            mel_dim=100,
        ),
        mel_spec_kwargs=MEL_SPEC_KWARGS,
        odeint_kwargs=ODEINT_KWARGS,
        vocab_char_map=vocab_char_map,
    )
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class F5FinetuneDataset(Dataset):
    """Dataset reading metadata.csv (audio_file|text_ipa) and loading WAV files as mel specs."""

    def __init__(self, dataset_path: str, mel_spectrogram=None):
        self.wavs_dir = os.path.join(dataset_path, "wavs")
        self.mels_dir = os.path.join(dataset_path, "mels_cache")
        metadata_path = os.path.join(dataset_path, "metadata.csv")

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")
        if not os.path.isdir(self.wavs_dir):
            raise FileNotFoundError(f"wavs/ directory not found at: {self.wavs_dir}")

        self.samples: list[tuple[str, str]] = []
        with open(metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 2:
                    continue
                audio_file, text_ipa = row[0].strip(), row[1].strip()
                if not audio_file or not text_ipa:
                    continue
                self.samples.append((audio_file, text_ipa))

        logger.info(f"Loaded {len(self.samples)} samples from {metadata_path}")

        # Pre-cache mel spectrograms to disk (one-time cost)
        if mel_spectrogram is not None:
            self._ensure_mel_cache(mel_spectrogram)

    def _ensure_mel_cache(self, mel_spectrogram) -> None:
        """Pre-compute mel spectrograms and save as .pt files."""
        os.makedirs(self.mels_dir, exist_ok=True)

        # Check if cache is already complete
        uncached = [
            i for i, (af, _) in enumerate(self.samples)
            if not os.path.isfile(os.path.join(self.mels_dir, af.replace(".wav", ".pt")))
        ]

        if not uncached:
            logger.info(f"Mel cache complete: {len(self.samples)} files in {self.mels_dir}")
            return

        logger.info(f"Pre-caching {len(uncached)} mel spectrograms to {self.mels_dir} ...")
        import soundfile as sf
        for i in tqdm(uncached, desc="Caching mels", unit="file"):
            audio_file, _ = self.samples[i]
            wav_path = os.path.join(self.wavs_dir, audio_file)
            mel_path = os.path.join(self.mels_dir, audio_file.replace(".wav", ".pt"))

            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 24000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

            wav_tensor = torch.from_numpy(audio).float()
            with torch.no_grad():
                mel = mel_spectrogram(wav_tensor.unsqueeze(0))  # [1, 100, T]
            mel = mel.squeeze(0).T  # [T, 100]
            torch.save(mel, mel_path)

        logger.info("Mel cache complete.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Return (mel [T, 100], text_ipa). Uses cached mel if available."""
        audio_file, text_ipa = self.samples[idx]

        # Try cached mel first
        mel_path = os.path.join(self.mels_dir, audio_file.replace(".wav", ".pt"))
        if os.path.isfile(mel_path):
            mel = torch.load(mel_path, map_location="cpu", weights_only=True)
            return mel, text_ipa

        # Fallback: load WAV (should not happen if cache is built)
        wav_path = os.path.join(self.wavs_dir, audio_file)
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        wav_tensor = torch.from_numpy(audio).float()
        return wav_tensor, text_ipa


def _compute_mel(wav_tensor: torch.Tensor, mel_spectrogram) -> torch.Tensor:
    """Compute mel spectrogram from a 1-D wav tensor. Returns [T, 100]."""
    # mel_spectrogram expects [B, N] or [N]
    with torch.no_grad():
        mel = mel_spectrogram(wav_tensor.unsqueeze(0))  # [1, 100, T]
    return mel.squeeze(0).T  # [T, 100]


class FrameBudgetCollator:
    """
    Dynamic batching collator: accumulates samples until either
    total_frames >= batch_size_per_gpu or sample count >= max_samples.
    Returns a single batch as (mels_padded, texts, lens).

    Expects items from dataset to be (mel [T, 100], text_ipa) when mel cache
    is available, or (wav_tensor [N], text_ipa) as fallback.
    """

    def __init__(self, batch_size_per_gpu: int, max_samples: int, mel_spectrogram=None):
        self.batch_size_per_gpu = batch_size_per_gpu
        self.max_samples = max_samples
        self.mel_spectrogram = mel_spectrogram

    def __call__(self, batch: list[tuple[torch.Tensor, str]]):
        selected_mels: list[torch.Tensor] = []
        selected_texts: list[str] = []
        total_frames = 0

        for tensor, text_ipa in batch:
            # If tensor is 2D [T, 100] it's already a mel; if 1D [N] it's a wav
            if tensor.ndim == 1 and self.mel_spectrogram is not None:
                mel = _compute_mel(tensor, self.mel_spectrogram)
            else:
                mel = tensor  # already [T, 100] from cache

            n_frames = mel.shape[0]

            if total_frames + n_frames > self.batch_size_per_gpu and selected_mels:
                break

            selected_mels.append(mel)
            selected_texts.append(text_ipa)
            total_frames += n_frames

            if len(selected_mels) >= self.max_samples:
                break

        lens = torch.tensor([m.shape[0] for m in selected_mels], dtype=torch.long)
        mels_padded = pad_sequence(selected_mels, batch_first=True)  # [B, T_max, 100]
        return mels_padded, selected_texts, lens


def build_dataloader(
    dataset: F5FinetuneDataset,
    batch_size_per_gpu: int,
    max_samples: int,
    mel_spectrogram,
    num_workers: int = 0,
) -> DataLoader:
    """Build DataLoader with FrameBudgetCollator."""
    collator = FrameBudgetCollator(
        batch_size_per_gpu=batch_size_per_gpu,
        max_samples=max_samples,
        mel_spectrogram=mel_spectrogram,
    )
    # batch_size here is the maximum pre-collation batch; collator trims it
    return DataLoader(
        dataset,
        batch_size=max_samples,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str,
    step: int,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
) -> str:
    """Save checkpoint in EMA-compatible format expected by load_checkpoint,
    plus optimizer/scheduler state for resume."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"model_{step:07d}.pt"
    ckpt_path = os.path.join(output_dir, filename)

    # Unwrap accelerate / DDP if necessary
    raw_model = model
    if hasattr(model, "module"):
        raw_model = model.module

    state = {
        "ema_model_state_dict": {
            "initted": torch.tensor(True),
            "step": torch.tensor(step),
            **{f"ema_model.{k}": v.cpu() for k, v in raw_model.state_dict().items()},
        },
        "global_step": step,
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest model_*.pt checkpoint in output_dir."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("model_") and f.endswith(".pt")]
    )
    if not ckpts:
        return None
    return os.path.join(output_dir, ckpts[-1])


def cleanup_old_checkpoints(output_dir: str, keep: int) -> None:
    """Keep only the last `keep` checkpoints, delete older ones."""
    if keep <= 0:
        return
    if not os.path.isdir(output_dir):
        return
    ckpts = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("model_") and f.endswith(".pt")]
    )
    if len(ckpts) <= keep:
        return
    for old_ckpt in ckpts[:-keep]:
        old_path = os.path.join(output_dir, old_ckpt)
        os.remove(old_path)
        logger.info(f"Removed old checkpoint: {old_path}")


# ---------------------------------------------------------------------------
# Learning rate scheduler (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, num_warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, total_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    cfg: dict,
    max_steps: int | None = None,
    resume: bool = False,
    no_mel_cache: bool = False,
    delete_mel_cache: bool = False,
    keep_checkpoints: int = 3,
) -> None:
    """Main training function."""
    # --- Accelerate setup ---
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError as exc:
        raise ImportError("accelerate is required for training. Run: pip install accelerate") from exc

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=cfg.get("grad_accumulation_steps", 2),
        log_with=None,  # wandb init handled manually below
    )

    set_seed(42)

    is_main = accelerator.is_main_process

    if is_main:
        logger.info("=== F5-TTS-TH-V2 Fine-tuning ===")
        logger.info(f"Config: {cfg}")
        if max_steps:
            logger.info(f"Short run mode: max_steps={max_steps}")

    # --- Resolve paths (relative to script location) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(script_dir, p))

    dataset_path = resolve_path(cfg["dataset_path"])
    output_dir = resolve_path(cfg["output_dir"])

    # --- Pretrained checkpoint ---
    ckpt_path, vocab_file = resolve_pretrained_checkpoint(cfg)

    # Override resolved vocab_file if explicitly given and exists
    if cfg.get("vocab_file") and os.path.isfile(resolve_path(cfg["vocab_file"])):
        vocab_file = resolve_path(cfg["vocab_file"])

    if is_main:
        logger.info(f"Checkpoint: {ckpt_path}")
        logger.info(f"Vocab: {vocab_file}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output: {output_dir}")

    # --- Build model ---
    model = build_model(vocab_file)

    # Load pretrained weights
    try:
        from f5_tts_th.utils_infer import load_checkpoint
        model = load_checkpoint(
            model,
            ckpt_path,
            device="cpu",       # load to CPU first; accelerate will move
            dtype=torch.float32,
            use_ema=True,       # pretrained checkpoint is always EMA format
        )
        if is_main:
            logger.info("Pretrained weights loaded successfully.")
    except Exception as exc:
        raise RuntimeError(f"Failed to load pretrained checkpoint: {exc}") from exc

    # --- Extract mel_spectrogram from the CFM model for dataset collation ---
    # CFM stores its MelSpectrogram as self.mel_spec
    mel_spectrogram = model.mel_spec.to("cpu")

    # --- Delete mel cache if requested ---
    if delete_mel_cache:
        import shutil
        mels_cache_dir = os.path.join(dataset_path, "mels_cache")
        if os.path.isdir(mels_cache_dir):
            if is_main:
                logger.info(f"Deleting mel cache: {mels_cache_dir}")
            shutil.rmtree(mels_cache_dir)
            if is_main:
                logger.info("Mel cache deleted.")

    # --- Dataset & DataLoader ---
    # Pass mel_spectrogram to dataset for one-time mel caching to disk (unless --no-mel-cache)
    cache_mel_spec = None if no_mel_cache else mel_spectrogram
    if no_mel_cache and is_main:
        logger.info("Mel caching disabled — computing on-the-fly (slower but saves ~150GB+ disk)")
    dataset = F5FinetuneDataset(dataset_path, mel_spectrogram=cache_mel_spec)
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size_per_gpu=cfg["batch_size_per_gpu"],
        max_samples=cfg["max_samples"],
        mel_spectrogram=mel_spectrogram,  # fallback for uncached samples
        num_workers=0,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # --- LR Scheduler ---
    steps_per_epoch = len(dataloader)
    total_epochs = cfg["epochs"]
    total_steps_estimate = steps_per_epoch * total_epochs // cfg.get("grad_accumulation_steps", 2)
    if max_steps:
        total_steps_estimate = max_steps

    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=cfg["num_warmup_updates"],
        total_steps=total_steps_estimate,
    )

    # --- Accelerate prepare ---
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # --- WandB (optional, main process only) ---
    wandb_run = None
    if is_main and cfg.get("wandb_project"):
        try:
            import wandb
            wandb_mode = cfg.get("wandb_mode", "offline")
            wandb_run = wandb.init(
                project=cfg["wandb_project"],
                mode=wandb_mode,
                config=cfg,
            )
            logger.info(f"WandB initialized (mode={wandb_mode}).")
        except Exception as exc:
            logger.warning(f"WandB init failed (continuing without logging): {exc}")

    # --- Resume from checkpoint ---
    global_step = 0
    start_epoch = 0

    if resume:
        resume_ckpt = find_latest_checkpoint(output_dir)
        if resume_ckpt and is_main:
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")
            try:
                ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
                global_step = ckpt.get("global_step", 0)
                start_epoch = ckpt.get("epoch", 0)
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                del ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Resumed: epoch={start_epoch}, global_step={global_step}")
            except (RuntimeError, EOFError, zipfile.BadZipFile) as exc:
                logger.warning(f"Checkpoint is corrupted, removing it: {resume_ckpt} ({exc})")
                os.remove(resume_ckpt)
                # Try the next most recent checkpoint
                resume_ckpt = find_latest_checkpoint(output_dir)
                if resume_ckpt:
                    logger.info(f"Falling back to: {resume_ckpt}")
                    ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
                    global_step = ckpt.get("global_step", 0)
                    start_epoch = ckpt.get("epoch", 0)
                    if "optimizer_state_dict" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    del ckpt
                    logger.info(f"Resumed: epoch={start_epoch}, global_step={global_step}")
                else:
                    logger.info("No valid checkpoint found, starting from scratch.")
        elif is_main:
            logger.info("No checkpoint found in output_dir, starting from scratch.")

    # --- Training loop ---
    save_per_updates = cfg.get("save_per_updates", 500)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    grad_accumulation_steps = cfg.get("grad_accumulation_steps", 2)
    log_interval = cfg.get("log_interval", 1)

    model.train()
    current_epoch = start_epoch

    if is_main:
        logger.info(f"Starting training: {total_epochs} epochs, {steps_per_epoch} steps/epoch")

    for epoch in range(start_epoch, total_epochs):
        current_epoch = epoch
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_start = time.time()

        for batch_idx, (mels, texts, lens) in enumerate(dataloader):
            # mels: [B, T_max, 100] — on accelerator device
            # texts: list[str]
            # lens: [B]

            with accelerator.accumulate(model):
                try:
                    loss, _cond, _pred = model(
                        inp=mels,    # [B, T, 100] mel spectrograms
                        text=texts,  # list of IPA strings
                        lens=lens,   # [B] actual mel frame lengths
                    )
                except Exception as exc:
                    logger.error(f"Forward pass failed at step {global_step}: {exc}")
                    raise

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss={loss.item()} at step {global_step}, skipping batch.")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Capture loss value before freeing tensors
            loss_val = loss.detach().item()
            del loss, _cond, _pred

            # Track loss

            epoch_loss_sum += loss_val
            epoch_batches += 1

            # Increment global step after each optimizer update
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                global_step += 1

                if is_main:
                    lr_current = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg["learning_rate"]
                    if global_step % log_interval == 0 or global_step == 1:
                        logger.info(
                            f"Epoch {epoch + 1}/{total_epochs} | "
                            f"Step {global_step} | "
                            f"Loss {loss_val:.4f} | "
                            f"LR {lr_current:.2e}"
                        )

                    if wandb_run:
                        try:
                            wandb_run.log({
                                "train/loss": loss_val,
                                "train/lr": lr_current,
                                "train/epoch": epoch + 1,
                                "train/global_step": global_step,
                            }, step=global_step)
                        except Exception:
                            pass  # Don't crash training on logging failure

                # --- Save checkpoint ---
                if global_step % save_per_updates == 0 and is_main:
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        output_dir,
                        global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                    )
                    cleanup_old_checkpoints(output_dir, keep_checkpoints)

                # --- Max steps early stop ---
                if max_steps and global_step >= max_steps:
                    if is_main:
                        logger.info(f"Reached max_steps={max_steps}, stopping.")
                    break

        # End of epoch — free VRAM cache to prevent progressive slowdown
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_main and epoch_batches > 0:
            avg_loss = epoch_loss_sum / epoch_batches
            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1} complete | "
                f"Avg Loss {avg_loss:.4f} | "
                f"Time {elapsed:.1f}s"
            )
            if wandb_run:
                try:
                    wandb_run.log({
                        "epoch/avg_loss": avg_loss,
                        "epoch/index": epoch + 1,
                    }, step=global_step)
                except Exception:
                    pass

        if max_steps and global_step >= max_steps:
            break

    # --- Final checkpoint ---
    if is_main:
        save_checkpoint(
            accelerator.unwrap_model(model),
            output_dir,
            global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=current_epoch,
        )
        logger.info("Training complete.")

        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune F5-TTS-TH-V2 on a prepared dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        dest="max_steps",
        help="Stop after this many optimizer steps (useful for quick tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--no-mel-cache",
        action="store_true",
        default=False,
        dest="no_mel_cache",
        help="Compute mel spectrograms on-the-fly instead of caching to disk. Saves ~150GB+ disk space.",
    )
    parser.add_argument(
        "--delete-mel-cache",
        action="store_true",
        default=False,
        dest="delete_mel_cache",
        help="Delete existing mels_cache/ directory before training to free disk space.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        dest="keep_checkpoints",
        help="Keep only the last N checkpoints to save disk space (0 = keep all).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config and merge with defaults."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **user_cfg}
    return cfg


def main() -> None:
    args = parse_args()

    # Resolve config path relative to cwd
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)

    train(
        cfg,
        max_steps=args.max_steps,
        resume=args.resume,
        no_mel_cache=args.no_mel_cache,
        delete_mel_cache=args.delete_mel_cache,
        keep_checkpoints=args.keep_checkpoints,
    )


if __name__ == "__main__":
    main()
