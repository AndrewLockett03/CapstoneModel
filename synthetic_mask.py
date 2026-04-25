def generate_synthetic_mask(
    n_frames: int,
    min_run: int = 1,
    max_run: int = 12,
    n_events: int = None,
) -> torch.Tensor:
    """
    Generate a random clipping mask for training, simulating what ClipDaT produces.
    Returns a bool tensor of shape (n_frames,), True = masked.

    Produces short contiguous runs of masked frames, matching the typical
    distribution of real clipping events (mostly brief, occasionally longer).
    """
    mask = torch.zeros(n_frames, dtype=torch.bool)

    if n_events is None:
        n_events = torch.randint(1, 6, (1,)).item()

    for _ in range(n_events):
        run_length = torch.randint(min_run, max_run + 1, (1,)).item()
        start = torch.randint(0, max(1, n_frames - run_length), (1,)).item()
        mask[start : start + run_length] = True

    return mask