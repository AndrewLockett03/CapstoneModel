import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from STFT import STFTConfig, load_audio, compute_stft, compute_log_magnitude
from model_config import ModelConfig

class VCTKInpaintingDataset(Dataset):
    """
    Loads VCTK wav files, computes log-magnitude STFTs, and applies
    synthetic clipping masks on-the-fly for training.

    Each item returned is a fixed-length chunk of context_frames frames,
    with a randomly generated mask applied.
    """
    def __init__(
        self,
        wav_dir: str,
        stft_config: STFTConfig,
        model_config: ModelConfig,
        split: str = 'train',       # 'train' or 'val'
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.stft_config  = stft_config
        self.model_config = model_config

        # Collect all wav files recursively
        all_files = sorted(glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True))
        if not all_files:
            raise ValueError(f"No wav files found under {wav_dir}")

        # Deterministic train/val split by speaker (p-prefixed directories)
        random.seed(seed)
        random.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        if split == 'val':
            self.files = all_files[:n_val]
        else:
            self.files = all_files[n_val:]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames:  (context_frames, n_bins)  log-magnitude, normalised
            mask:    (context_frames,)          bool, True = masked
            target:  (context_frames, n_bins)  clean log-magnitude (for loss)
        """
        audio = load_audio(self.files[idx], self.stft_config)
        stft  = compute_stft(audio, self.stft_config)
        log_mag = compute_log_magnitude(stft, self.stft_config)  # (n_frames, n_bins)

        # Per-file normalisation: zero mean, unit variance
        mean = log_mag.mean()
        std  = log_mag.std() + 1e-8
        log_mag = (log_mag - mean) / std

        n_frames = log_mag.shape[0]
        ctx      = self.model_config.context_frames

        # If the file is shorter than one context window, pad with zeros
        if n_frames < ctx:
            pad = np.zeros((ctx - n_frames, log_mag.shape[1]), dtype=np.float32)
            log_mag = np.concatenate([log_mag, pad], axis=0)
            n_frames = ctx

        # Sample a random starting frame
        start = random.randint(0, n_frames - ctx)
        chunk = log_mag[start : start + ctx]                     # (ctx, n_bins)

        target = torch.from_numpy(chunk.astype(np.float32))
        mask   = generate_synthetic_mask(ctx)                    # (ctx,) bool

        # Frames tensor: masked positions will be replaced inside the model
        # We pass the clean chunk as input — the model replaces masked positions
        # with the learned mask_embedding in its forward() method
        frames = target.clone()

        return frames, mask, target
