def inpaint(
    model: SpectrogramInpainter,
    log_magnitude: torch.Tensor,    # (n_frames, n_bins)  full file or chunk
    mask: torch.Tensor,              # (n_frames,)          from ClipDaT
    config: ModelConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Run inference over a full spectrogram using a sliding context window,
    stitching results together at the end.

    For each masked region, we centre a context window of config.context_frames
    around it so the model has equal amounts of left and right context.
    Returns a copy of log_magnitude with masked frames replaced by predictions.
    """
    model.eval()
    result = log_magnitude.clone()
    n_frames = log_magnitude.size(0)

    # Find contiguous masked runs
    masked_indices = mask.nonzero(as_tuple=True)[0]
    if len(masked_indices) == 0:
        return result

    # Group into contiguous runs
    runs = []
    run_start = masked_indices[0].item()
    run_end   = masked_indices[0].item()
    for idx in masked_indices[1:]:
        idx = idx.item()
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = run_end = idx
    runs.append((run_start, run_end))

    with torch.no_grad():
        for run_start, run_end in runs:
            run_centre = (run_start + run_end) // 2
            half       = config.context_frames // 2

            # Centre the window on the masked run, clamped to file boundaries
            win_start  = max(0, run_centre - half)
            win_end    = min(n_frames, win_start + config.context_frames)
            win_start  = max(0, win_end - config.context_frames)  # re-align if clamped

            chunk      = log_magnitude[win_start:win_end].unsqueeze(0).to(device)
            chunk_mask = mask[win_start:win_end].unsqueeze(0).to(device)

            predictions = model(chunk, chunk_mask)  # (1, context_frames, n_bins)

            # Write only the masked frames back into the result
            local_mask = chunk_mask.squeeze(0)
            result[win_start:win_end][local_mask] = (
                predictions.squeeze(0)[local_mask].cpu()
            )

    return result