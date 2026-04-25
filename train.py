import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Normalisation stats helper (optional — for tracking across an epoch)
# ---------------------------------------------------------------------------

class RunningStats:
    """Accumulates mean and std of a scalar quantity over batches."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model:        SpectrogramInpainter,
    stft_config:  STFTConfig,
    model_config: ModelConfig,
    wav_dir:      str,
    n_epochs:     int   = 50,
    batch_size:   int   = 32,
    lr:           float = 3e-4,
    checkpoint_dir: str = './checkpoints',
    resume_from:  str   = None,        # path to a checkpoint to resume from
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    model = model.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Data ---
    train_dataset = VCTKInpaintingDataset(wav_dir, stft_config, model_config, split='train')
    val_dataset   = VCTKInpaintingDataset(wav_dir, stft_config, model_config, split='val')

    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader    = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    print(f"Train files: {len(train_dataset)}  |  Val files: {len(val_dataset)}")

    # --- Optimiser & scheduler ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    start_epoch  = 0
    best_val_loss = float('inf')

    # --- Optional resume ---
    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    # --- Epoch loop ---
    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_stats = RunningStats()

        for batch_idx, (frames, mask, target) in enumerate(train_loader):
            frames  = frames.to(device)   # (B, ctx, n_bins)
            mask    = mask.to(device)     # (B, ctx)
            target  = target.to(device)   # (B, ctx, n_bins)

            optimizer.zero_grad()
            predictions = model(frames, mask)               # (B, ctx, n_bins)
            loss = inpainting_loss(predictions, target, mask)
            loss.backward()

            # Gradient clipping — important for transformer stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_stats.update(loss.item())

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs} "
                    f"| Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        # --- Validation ---
        val_loss = _validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{n_epochs} complete "
            f"| Train loss {train_stats.mean:.4f} "
            f"| Val loss {val_loss:.4f} "
            f"| LR {scheduler.get_last_lr()[0]:.2e}"
        )

        # --- Checkpointing ---
        checkpoint = {
            'epoch':          epoch,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
            'best_val_loss':  best_val_loss,
            'model_config':   model_config,
            'stft_config':    stft_config,
        }

        # Always save latest
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))
            print(f"  -> New best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def _validate(
    model:      SpectrogramInpainter,
    val_loader: DataLoader,
    device:     torch.device,
) -> float:
    model.eval()
    stats = RunningStats()

    for frames, mask, target in val_loader:
        frames  = frames.to(device)
        mask    = mask.to(device)
        target  = target.to(device)

        predictions = model(frames, mask)
        loss = inpainting_loss(predictions, target, mask)
        stats.update(loss.item())

    return stats.mean


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    stft_config  = STFTConfig()
    model_config = ModelConfig()
    model        = SpectrogramInpainter(model_config)

    VCTK_WAV_DIR = "/root/.cache/kagglehub/datasets/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48"

    train(
        model=model,
        stft_config=stft_config,
        model_config=model_config,
        wav_dir=VCTK_WAV_DIR,
        n_epochs=50,
        batch_size=32,
        lr=3e-4,
        checkpoint_dir='./checkpoints',
        resume_from=None,   # set to './checkpoints/latest.pt' to resume
    )