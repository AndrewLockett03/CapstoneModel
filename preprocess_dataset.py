from pathlib import Path
from tqdm import tqdm
from STFT import STFTConfig, load_audio, compute_stft, compute_log_magnitude, get_frame_indices_for_samples
import numpy as np
from preprocess import induce_clipping, clip_detection
import glob
import os

def preprocess_dataset(
    wav_dir:     str,
    output_dir:  str,
    stft_config: STFTConfig,
):
    """
    For each wav file, run the full preprocessing chain and save the result
    as a compressed .npz file. Skips files that already have a cached result.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True))
    print(f"Preprocessing {len(wav_files)} files into {output_dir}")

    for wav_file in tqdm(wav_files):
        # Derive a stable output filename from the relative path
        rel      = Path(wav_file).relative_to(wav_dir)
        out_file = (output_path / rel).with_suffix('.npz')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue  # already cached

        clean_audio   = load_audio(wav_file, stft_config)
        clipped_audio = induce_clipping(clean_audio)
        clip_events   = clip_detection(clipped_audio)

        clipped_stft    = compute_stft(clipped_audio, stft_config)
        clean_stft      = compute_stft(clean_audio,   stft_config)
        n_frames        = clipped_stft.shape[0]

        clipped_log_mag = compute_log_magnitude(clipped_stft, stft_config)
        clean_log_mag   = compute_log_magnitude(clean_stft,   stft_config)

        mean = clean_log_mag.mean()
        std  = clean_log_mag.std() + 1e-8
        clipped_log_mag = (clipped_log_mag - mean) / std
        clean_log_mag   = (clean_log_mag   - mean) / std

        # Build frame mask
        frame_mask = np.zeros(n_frames, dtype=bool)
        for sample_start, sample_end in clip_events:
            fs, fe = get_frame_indices_for_samples(sample_start, sample_end, stft_config)
            fe = min(fe, n_frames - 1)
            if fs <= fe:
                frame_mask[fs : fe + 1] = True

        np.savez_compressed(
            out_file,
            clipped_log_mag = clipped_log_mag.astype(np.float32),
            clean_log_mag   = clean_log_mag.astype(np.float32),
            frame_mask      = frame_mask,
        )

    print("Preprocessing complete.")