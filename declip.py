import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from model_config import ModelConfig  # adjust import to match your file structure
from STFT import STFTConfig, load_audio
from spectrogram_inpainter import SpectrogramInpainter


def propagate_phase_bidirectional(stft, mask_start, mask_end, hop, fft_size):
    """Bidirectional phase propagation into a masked region."""
    n_bins = stft.shape[1]
    mask_len = mask_end - mask_start + 1
    bins = np.arange(n_bins)
    delta_phi_expected = 2 * np.pi * bins * hop / fft_size

    result = stft.copy()

    def _wrap(phase):
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def _circular_blend(phi_a, phi_b, alpha):
        return phi_a + alpha * _wrap(phi_b - phi_a)

    ref_left = mask_start - 1
    if ref_left >= 1:
        dev_left = _wrap(
            (np.angle(stft[ref_left]) - np.angle(stft[ref_left - 1])) - delta_phi_expected
        )
    else:
        dev_left = np.zeros(n_bins)

    ref_right = mask_end + 1
    if ref_right < stft.shape[0] - 1:
        dev_right = _wrap(
            (np.angle(stft[ref_right + 1]) - np.angle(stft[ref_right])) - delta_phi_expected
        )
    else:
        dev_right = np.zeros(n_bins)

    for n, frame_idx in enumerate(range(mask_start, mask_end + 1)):
        alpha = n / (mask_len - 1) if mask_len > 1 else 0.5

        steps_from_left = n + 1
        phi_left = np.angle(stft[ref_left]) + steps_from_left * (delta_phi_expected + dev_left)

        steps_from_right = mask_len - n
        phi_right = np.angle(stft[ref_right]) - steps_from_right * (delta_phi_expected + dev_right)

        phi_blended = _circular_blend(phi_left, phi_right, alpha)
        mag = np.abs(result[frame_idx])
        result[frame_idx] = mag * np.exp(1j * phi_blended)

    return result


def run_inference(
    model:       'SpectrogramInpainter',
    log_mag:     np.ndarray,          # (n_frames, n_bins) normalised
    frame_mask:  np.ndarray,          # (n_frames,) bool
    model_config:'ModelConfig',
    device:      torch.device,
) -> np.ndarray:
    """
    Slide a context window over the full spectrogram, inpainting each
    masked run. Returns a (n_frames, n_bins) log-magnitude array with
    masked frames replaced by model predictions, in the normalised scale.
    """
    model.eval()
    result   = log_mag.copy()
    n_frames = log_mag.shape[0]
    ctx      = model_config.context_frames

    # Find contiguous masked runs
    masked_indices = np.where(frame_mask)[0]
    if len(masked_indices) == 0:
        print("  Warning: no clipping events detected in this file.")
        return result

    runs = []
    run_start = run_end = masked_indices[0]
    for idx in masked_indices[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = run_end = idx
    runs.append((run_start, run_end))

    print(f"  Inpainting {len(runs)} clipping event(s) across {len(masked_indices)} masked frames")

    with torch.no_grad():
        for run_start, run_end in runs:
            run_centre = (run_start + run_end) // 2
            half       = ctx // 2

            win_start = max(0, run_centre - half)
            win_end   = min(n_frames, win_start + ctx)
            win_start = max(0, win_end - ctx)

            chunk      = torch.from_numpy(log_mag[win_start:win_end]).unsqueeze(0).to(device)
            chunk_mask = torch.from_numpy(frame_mask[win_start:win_end]).unsqueeze(0).to(device)

            predictions = model(chunk, chunk_mask)   # (1, ctx, n_bins)

            local_mask = chunk_mask.squeeze(0)
            result[win_start:win_end][local_mask.cpu().numpy()] = (
                predictions.squeeze(0)[local_mask].cpu().numpy()
            )

    return result


def test(
    checkpoint_path: str,
    audio_path:      str,
    output_dir:      str = './test_output',
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.serialization.add_safe_globals([ModelConfig, STFTConfig])

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint['model_config']
    stft_config  = checkpoint['stft_config']

    model = SpectrogramInpainter(model_config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"  Trained for {checkpoint['epoch'] + 1} epoch(s), "
          f"best val loss {checkpoint['best_val_loss']:.4f}")

    # --- Load and process audio ---
    print(f"\nProcessing: {audio_path}")
    clean_audio   = load_audio(audio_path, stft_config)
    clipped_audio = induce_clipping(clean_audio)
    clip_events   = clip_detection(clipped_audio)

    print(f"  Samples:       {len(clean_audio)}")
    print(f"  Clip events:   {len(clip_events)}")
    if clip_events:
        total_clipped = sum(e - s + 1 for s, e in clip_events)
        print(f"  Clipped samples: {total_clipped} ({100 * total_clipped / len(clean_audio):.2f}%)")

    # --- Build STFTs ---
    clipped_stft = compute_stft(clipped_audio, stft_config)
    clean_stft   = compute_stft(clean_audio,   stft_config)
    n_frames     = clipped_stft.shape[0]

    clipped_log_mag = compute_log_magnitude(clipped_stft, stft_config)
    clean_log_mag   = compute_log_magnitude(clean_stft,   stft_config)

    # Normalise with clean stats (matching training convention)
    mean = clean_log_mag.mean()
    std  = clean_log_mag.std() + 1e-8
    clipped_log_mag_norm = (clipped_log_mag - mean) / std

    # --- Build frame mask from ClipDaT events ---
    frame_mask = np.zeros(n_frames, dtype=bool)
    for sample_start, sample_end in clip_events:
        fs, fe = get_frame_indices_for_samples(sample_start, sample_end, stft_config)
        fe = min(fe, n_frames - 1)
        if fs <= fe:
            frame_mask[fs : fe + 1] = True

    # --- Run model ---
    print("\nRunning inference...")
    inpainted_log_mag_norm = run_inference(
        model, clipped_log_mag_norm, frame_mask, model_config, device
    )

    # Denormalise back to log-magnitude scale
    inpainted_log_mag = inpainted_log_mag_norm * std + mean

    # --- Phase propagation into masked runs ---
    print("Propagating phase...")

    # Start from the clipped STFT so unmasked frames retain original phase
    output_stft = clipped_stft.copy()

    # Replace masked frame magnitudes with inpainted values
    inpainted_mag = log_magnitude_to_magnitude(inpainted_log_mag, stft_config)
    output_stft[frame_mask] = (
        inpainted_mag[frame_mask] * np.exp(1j * np.angle(clipped_stft[frame_mask]))
    )

    # Now run bidirectional phase propagation over each masked run
    masked_indices = np.where(frame_mask)[0]
    if len(masked_indices) > 0:
        runs = []
        run_start = run_end = masked_indices[0]
        for idx in masked_indices[1:]:
            if idx == run_end + 1:
                run_end = idx
            else:
                runs.append((run_start, run_end))
                run_start = run_end = idx
        runs.append((run_start, run_end))

        for run_start, run_end in runs:
            # Guard against runs that touch the very start or end of the file
            if run_start < 1 or run_end >= n_frames - 1:
                continue
            output_stft = propagate_phase_bidirectional(
                output_stft, run_start, run_end,
                stft_config.hop_length, stft_config.n_fft
            )

    # --- ISTFT -> waveform ---
    output_audio = istft(output_stft, stft_config, orig_len=len(clean_audio))

    # --- Save all three versions ---
    stem = Path(audio_path).stem
    sr   = stft_config.sample_rate

    original_out = output_path / f"{stem}_original.wav"
    clipped_out  = output_path / f"{stem}_clipped.wav"
    restored_out = output_path / f"{stem}_restored.wav"

    sf.write(original_out, clean_audio,   sr)
    sf.write(clipped_out,  clipped_audio, sr)
    sf.write(restored_out, output_audio,  sr)

    print(f"\nSaved:")
    print(f"  Original : {original_out}")
    print(f"  Clipped  : {clipped_out}")
    print(f"  Restored : {restored_out}")

    # --- Simple quality metrics in log-magnitude domain ---
    # Compare only at masked frames, since unmasked frames are untouched
    if frame_mask.any():
        clean_masked    = clean_log_mag[frame_mask]
        clipped_masked  = clipped_log_mag[frame_mask]
        restored_masked = inpainted_log_mag[frame_mask]

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        print(f"\nLog-magnitude MAE at clipped frames:")
        print(f"  Clipped  vs clean:   {mae(clipped_masked,  clean_masked):.4f}  (baseline — no restoration)")
        print(f"  Restored vs clean:   {mae(restored_masked, clean_masked):.4f}  (model output)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test declipping model on a single audio file.")
    parser.add_argument('--checkpoint', required=True, help="Path to best.pt checkpoint")
    parser.add_argument('--audio',      required=True, help="Path to input wav file")
    parser.add_argument('--output_dir', default='./test_output', help="Directory to save outputs")
    args = parser.parse_args()

    test(
        checkpoint_path = args.checkpoint,
        audio_path      = args.audio,
        output_dir      = args.output_dir,
    )