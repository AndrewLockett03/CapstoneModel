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
    def __init__(
        self,
        preprocessed_dir: str,
        model_config:     ModelConfig,
        split:            str   = 'train',
        val_fraction:     float = 0.1,
        seed:             int   = 42,
    ):
        self.model_config = model_config

        all_files = sorted(glob.glob(
            os.path.join(preprocessed_dir, '**', '*.npz'), recursive=True
        ))
        if not all_files:
            raise ValueError(f"No preprocessed .npz files found under {preprocessed_dir}. "
                             f"Run preprocess_dataset() first.")

        random.seed(seed)
        random.shuffle(all_files)
        n_val        = max(1, int(len(all_files) * val_fraction))
        self.files   = all_files[:n_val] if split == 'val' else all_files[n_val:]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx  = self.model_config.context_frames
        data = np.load(self.files[idx])

        clipped_log_mag = data['clipped_log_mag']   # (n_frames, n_bins)
        clean_log_mag   = data['clean_log_mag']     # (n_frames, n_bins)
        frame_mask      = data['frame_mask']        # (n_frames,)
        n_frames        = clipped_log_mag.shape[0]

        # Pad if necessary
        if n_frames < ctx:
            pad_len         = ctx - n_frames
            z               = np.zeros((pad_len, clipped_log_mag.shape[1]), dtype=np.float32)
            clipped_log_mag = np.concatenate([clipped_log_mag, z],                        axis=0)
            clean_log_mag   = np.concatenate([clean_log_mag,   z],                        axis=0)
            frame_mask      = np.concatenate([frame_mask, np.zeros(pad_len, dtype=bool)], axis=0)
            n_frames        = ctx

        # Sample a window that contains at least one masked frame
        attempts = 0
        while attempts < 10:
            start      = random.randint(0, n_frames - ctx)
            mask_chunk = frame_mask[start : start + ctx]
            if mask_chunk.any():
                break
            attempts += 1

        clipped_chunk = clipped_log_mag[start : start + ctx]
        clean_chunk   = clean_log_mag  [start : start + ctx]

        return (
            torch.from_numpy(clipped_chunk),
            torch.from_numpy(mask_chunk),
            torch.from_numpy(clean_chunk),
        )
