import torch
import torch.nn.functional as F


def inpainting_loss(
    predictions: torch.Tensor,   # (batch, n_frames, n_bins)
    targets: torch.Tensor,        # (batch, n_frames, n_bins)
    mask: torch.Tensor,           # (batch, n_frames) bool
) -> torch.Tensor:
    """
    Compute combined L1 + MSE loss over masked frames only.
    L1 reduces spectrogram blurring, MSE penalises large errors more heavily.
    """
    pred_masked   = predictions[mask]   # (n_masked_frames, n_bins)
    target_masked = targets[mask]       # (n_masked_frames, n_bins)

    loss_l1  = F.l1_loss(pred_masked, target_masked)
    loss_mse = F.mse_loss(pred_masked, target_masked)

    return loss_l1 + loss_mse