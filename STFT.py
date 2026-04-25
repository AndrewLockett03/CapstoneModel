import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Optional


@dataclass
class STFTConfig:
    sample_rate:  int   = 48000
    n_fft:        int   = 512    # ~11.6ms window at 44.1kHz
    hop_length:   int   = 256    # 50% overlap
    window:       str   = 'hann'
    log_floor:    float = 1e-5   # ~100dB dynamic range


def load_audio(path: str, config: STFTConfig) -> np.ndarray:
    """Load audio file, resampling to target sample rate if necessary."""
    audio, sr = librosa.load(path, sr=config.sample_rate, mono=True)
    return audio


def compute_stft(audio: np.ndarray, config: STFTConfig) -> np.ndarray:
    """
    Compute the full complex STFT.
    Returns complex array of shape (n_frames, n_bins) where
    n_bins = n_fft // 2 + 1  (single-sided, DC to Nyquist).
    """
    window = librosa.filters.get_window(config.window, config.n_fft)

    stft = librosa.stft(
        audio,
        n_fft=config.n_fft, # Corrected: Changed from config.hop_length to config.n_fft
        hop_length=config.hop_length,
        window=window,
        center=True,       # pad signal so frame 0 is centered on sample 0
    )
    # librosa returns (n_bins, n_frames); transpose to (n_frames, n_bins)
    return stft.T


def compute_log_magnitude(stft: np.ndarray, config: STFTConfig) -> np.ndarray:
    """
    Compute log-magnitude from complex STFT.

    Steps:
      1. Take absolute value  -> linear magnitude, shape (n_frames, n_bins)
      2. Clip to log_floor    -> prevents log(0) and sets noise floor
      3. Take natural log     -> compresses dynamic range

    Returns real array of shape (n_frames, n_bins).
    """
    magnitude = np.abs(stft)
    magnitude_clipped = np.maximum(magnitude, config.log_floor)
    log_magnitude = np.log(magnitude_clipped)
    return log_magnitude


def log_magnitude_to_magnitude(log_magnitude: np.ndarray, config: STFTConfig) -> np.ndarray:
    """Invert the log step. Values that were below log_floor will be restored to log_floor."""
    return np.exp(log_magnitude)


def compute_phase(stft: np.ndarray) -> np.ndarray:
    """Extract phase from complex STFT. Shape (n_frames, n_bins), values in [-π, π]."""
    return np.angle(stft)


def recombine(log_magnitude: np.ndarray, phase: np.ndarray, config: STFTConfig) -> np.ndarray:
    """
    Reconstruct complex STFT from log-magnitude and phase.
    Use this when you want to do ISTFT after inpainting.
    """
    magnitude = log_magnitude_to_magnitude(log_magnitude, config)
    return magnitude * np.exp(1j * phase)


def istft(stft: np.ndarray, config: STFTConfig, orig_len: Optional[int] = None) -> np.ndarray:
    """
    Invert complex STFT back to waveform.
    Expects input of shape (n_frames, n_bins) — transposes internally for librosa.
    """
    window = librosa.filters.get_window(config.window, config.n_fft)
    return librosa.istft(
        stft.T,            # librosa expects (n_bins, n_frames)
        hop_length=config.hop_length,
        window=window,
        center=True,
        length=orig_len,   # Added length parameter
    )


def get_frame_indices_for_samples(
    sample_start: int,
    sample_end: int,
    config: STFTConfig
) -> tuple[int, int]:
    """
    Convert a sample-domain clipping interval [sample_start, sample_end]
    to the range of STFT frames that overlap it.

    Returns (frame_start, frame_end) inclusive.
    With center=True, frame t is centered at sample t * hop_length.
    A frame covers [t*hop - n_fft//2, t*hop + n_fft//2].
    """
    half_window = config.n_fft // 2

    # Earliest frame whose window reaches sample_start
    frame_start = max(0, (sample_start - half_window) // config.hop_length)

    # Latest frame whose window reaches sample_end
    frame_end = (sample_end + half_window) // config.hop_length

    return frame_start, frame_end


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = STFTConfig()

    # --- Synthesize a test signal (two sine waves) ---
    #duration = 1.0  # seconds
    #t = np.linspace(0, duration, int(config.sample_rate * duration), endpoint=False)
    #audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1760 * t)
    audio = load_audio("/root/.cache/kagglehub/datasets/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_001.wav", config)

    # --- Forward pass ---
    stft        = compute_stft(audio, config)
    log_mag     = compute_log_magnitude(stft, config)
    phase       = compute_phase(stft)

    print(f"Audio samples:       {len(audio)}")
    print(f"STFT shape:          {stft.shape}     (n_frames, n_bins)")
    print(f"Log-magnitude shape: {log_mag.shape}")
    print(f"Log-mag range:       [{log_mag.min():.2f}, {log_mag.max():.2f}]")
    print(f"Frequency resolution:{config.sample_rate / config.n_fft:.1f} Hz/bin")
    print(f"Time resolution:     {config.hop_length / config.sample_rate * 1000:.1f} ms/frame")

    # --- Simulate what happens after inpainting: recombine and invert ---
    stft_reconstructed = recombine(log_mag, phase, config)
    audio_reconstructed = istft(stft_reconstructed, config, orig_len=len(audio))

    # Trim to original length (ISTFT may add a few samples of padding)
    # This line is now redundant as length is passed to istft.
    # audio_reconstructed = audio_reconstructed[:len(audio)]
    reconstruction_error = np.max(np.abs(audio - audio_reconstructed))
    print(f"Max reconstruction error (round-trip): {reconstruction_error:.2e}")

    # --- Show how to map a clipping event to frame indices ---
    # Suppose ClipDaT found clipping at samples 10000-10200
    clip_start, clip_end = 10000, 10200
    f_start, f_end = get_frame_indices_for_samples(clip_start, clip_end, config)
    print(f"\nClipping at samples [{clip_start}, {clip_end}]")
    print(f"  -> masks frames   [{f_start}, {f_end}]  ({f_end - f_start + 1} frames)")