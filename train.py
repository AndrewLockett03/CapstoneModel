import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from spectrogram_inpainter import SpectrogramInpainter
from STFT import STFTConfig
from model_config import ModelConfig
from dataset import VCTKInpaintingDataset
from inpainting_loss import inpainting_loss
from preprocess_dataset import preprocess_dataset


# ---------------------------------------------------------------------------
# Training loop (unchanged interface, updated semantics)
# ---------------------------------------------------------------------------

class RunningStats:
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


def train(
    model:            SpectrogramInpainter,
    stft_config:      STFTConfig,           # still needed for checkpoint saving
    model_config:     ModelConfig,
    preprocessed_dir: str,                  # replaces wav_dir
    n_epochs:         int   = 50,
    batch_size:       int   = 32,
    lr:               float = 3e-4,
    checkpoint_dir:   str   = './checkpoints',
    resume_from:      str   = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    model = model.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # train_dataset = VCTKInpaintingDataset(wav_dir, stft_config, model_config, split='train')
    # val_dataset   = VCTKInpaintingDataset(wav_dir, stft_config, model_config, split='val')

    train_dataset = VCTKInpaintingDataset(preprocessed_dir, model_config, split='train')
    val_dataset   = VCTKInpaintingDataset(preprocessed_dir, model_config, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    print(f"Train files: {len(train_dataset)}  |  Val files: {len(val_dataset)}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    start_epoch   = 0
    best_val_loss = float('inf')

    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_stats = RunningStats()

        for batch_idx, (clipped_frames, mask, clean_frames) in enumerate(train_loader):
            # clipped_frames: (B, ctx, n_bins) — distorted input to the model
            # mask:           (B, ctx)          — True where clipping was detected
            # clean_frames:   (B, ctx, n_bins)  — reconstruction target
            clipped_frames = clipped_frames.to(device)
            mask           = mask.to(device)
            clean_frames   = clean_frames.to(device)

            optimizer.zero_grad()
            predictions = model(clipped_frames, mask)

            # Loss is between model output and CLEAN frames at masked positions only
            loss = inpainting_loss(predictions, clean_frames, mask)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_stats.update(loss.item())

            if batch_idx % 50 == 0:
                # Report what fraction of frames in this batch were masked
                mask_rate = mask.float().mean().item()
                print(
                    f"Epoch {epoch+1}/{n_epochs} "
                    f"| Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f} "
                    f"| Mask rate {mask_rate:.3f}"
                )

        val_loss = _validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{n_epochs} complete "
            f"| Train loss {train_stats.mean:.4f} "
            f"| Val loss {val_loss:.4f} "
            f"| LR {scheduler.get_last_lr()[0]:.2e}"
        )

        checkpoint = {
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config':  model_config,
            'stft_config':   stft_config,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))

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

    for clipped_frames, mask, clean_frames in val_loader:
        clipped_frames = clipped_frames.to(device)
        mask           = mask.to(device)
        clean_frames   = clean_frames.to(device)

        predictions = model(clipped_frames, mask)
        loss = inpainting_loss(predictions, clean_frames, mask)
        stats.update(loss.item())

    return stats.mean


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    stft_config  = STFTConfig()
    model_config = ModelConfig()
    model        = SpectrogramInpainter(model_config)

    VCTK_WAV_DIR = "./.cache/kagglehub/datasets/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48"
    PREPROCESSED_DIR = "./vctk_preprocessed"

    # Run once — safe to re-run, skips already-cached files
    preprocess_dataset(VCTK_WAV_DIR, PREPROCESSED_DIR, STFTConfig())

    train(
    model            = model,
    stft_config      = stft_config,
    model_config     = model_config,
    preprocessed_dir = PREPROCESSED_DIR,   # replaces wav_dir=VCTK_WAV_DIR
    n_epochs         = 50,
    batch_size       = 32,
    lr               = 3e-4,
    checkpoint_dir   = './checkpoints',
    resume_from      = None,
)