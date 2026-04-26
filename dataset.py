import glob
import os
import random
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from STFT import STFTConfig, load_audio, compute_stft, compute_log_magnitude, get_frame_indices_for_samples
from model_config import ModelConfig
from preprocess import induce_clipping, clip_detection


class VCTKInpaintingDataset(Dataset):
    """
    For each audio file:
      1. Load clean audio                          -> target
      2. Induce clipping at 5%                     -> clipped input
      3. Run ClipDaT on clipped audio              -> sample-domain clip events
      4. Convert clip events to STFT frame mask    -> mask
      5. Compute log-magnitude STFTs of both       -> input frames, target frames
      6. Normalise using clean file stats           -> consistent scale

    Returns (clipped_frames, mask, clean_frames) per chunk.
    """
    def __init__(
        self,
        wav_dir:      str,
        stft_config:  STFTConfig,
        model_config: ModelConfig,
        split:        str   = 'train',
        val_fraction: float = 0.1,
        seed:         int   = 42,
    ):
        self.stft_config  = stft_config
        self.model_config = model_config

        all_files = sorted(glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True))
        if not all_files:
            raise ValueError(f"No wav files found under {wav_dir}")

        random.seed(seed)
        random.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == 'val' else all_files[n_val:]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.model_config.context_frames

        # 1. Load clean audio
        clean_audio = load_audio(self.files[idx], self.stft_config)

        # 2. Induce clipping
        clipped_audio = induce_clipping(clean_audio)

        # 3. Detect clip events in sample domain
        clip_events = clip_detection(clipped_audio)

        # 4. Convert sample-domain events to a boolean STFT frame mask
        #    We need to know n_frames first, so compute the clipped STFT
        clipped_stft    = compute_stft(clipped_audio, self.stft_config)
        clean_stft      = compute_stft(clean_audio,   self.stft_config)
        n_frames        = clipped_stft.shape[0]

        frame_mask = self._build_frame_mask(clip_events, n_frames)

        # 5. Log-magnitude of both
        clipped_log_mag = compute_log_magnitude(clipped_stft, self.stft_config)  # (n_frames, n_bins)
        clean_log_mag   = compute_log_magnitude(clean_stft,   self.stft_config)  # (n_frames, n_bins)

        # 6. Normalise — use clean file stats so target and input are on the same scale
        mean = clean_log_mag.mean()
        std  = clean_log_mag.std() + 1e-8
        clipped_log_mag = (clipped_log_mag - mean) / std
        clean_log_mag   = (clean_log_mag   - mean) / std

        # 7. Pad if shorter than one context window
        if n_frames < ctx:
            pad_len = ctx - n_frames
            clipped_log_mag = np.concatenate(
                [clipped_log_mag, np.zeros((pad_len, clipped_log_mag.shape[1]), dtype=np.float32)], axis=0
            )
            clean_log_mag = np.concatenate(
                [clean_log_mag, np.zeros((pad_len, clean_log_mag.shape[1]), dtype=np.float32)], axis=0
            )
            frame_mask = np.concatenate([frame_mask, np.zeros(pad_len, dtype=bool)], axis=0)
            n_frames = ctx

        # 8. Sample a random context window
        start = random.randint(0, n_frames - ctx)
        clipped_chunk = clipped_log_mag[start : start + ctx]   # (ctx, n_bins)
        clean_chunk   = clean_log_mag  [start : start + ctx]   # (ctx, n_bins)
        mask_chunk    = frame_mask     [start : start + ctx]    # (ctx,)

        # If the sampled window has no masked frames, the loss will be zero
        # and the batch contributes nothing useful. Re-sample until we get
        # at least one masked frame — with a fallback to avoid infinite loops
        # on files where clipping didn't produce any detectable events.
        attempts = 0
        while not mask_chunk.any() and attempts < 10:
            start         = random.randint(0, n_frames - ctx)
            clipped_chunk = clipped_log_mag[start : start + ctx]
            clean_chunk   = clean_log_mag  [start : start + ctx]
            mask_chunk    = frame_mask     [start : start + ctx]
            attempts     += 1

        return (
            torch.from_numpy(clipped_chunk.astype(np.float32)),
            torch.from_numpy(mask_chunk),
            torch.from_numpy(clean_chunk.astype(np.float32)),
        )

    def _build_frame_mask(
        self,
        clip_events: List[Tuple[int, int]],
        n_frames:    int,
    ) -> np.ndarray:
        """
        Convert a list of sample-domain clipping events to a boolean
        array of shape (n_frames,) in the STFT frame domain.
        """
        mask = np.zeros(n_frames, dtype=bool)
        for sample_start, sample_end in clip_events:
            frame_start, frame_end = get_frame_indices_for_samples(
                sample_start, sample_end, self.stft_config
            )
            # Clamp to valid frame range
            frame_end = min(frame_end, n_frames - 1)
            if frame_start <= frame_end:
                mask[frame_start : frame_end + 1] = True
        return mask


