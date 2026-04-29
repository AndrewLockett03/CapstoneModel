from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_bins:        int   = 257      # n_fft // 2 + 1
    d_model:       int   = 256      # transformer internal dimension
    d_ff:          int   = 1024     # feed-forward network hidden dimension
    dropout:       float = 0.1
    max_frames:    int   = 128      # maximum context window in frames
    context_frames: int  = 64      # how many frames to process at once at inference