class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds a fixed position-dependent signal to each frame embedding so the
    model knows the temporal order of frames.
    """
    def __init__(self, d_model: int, max_frames: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the encoding matrix once and register as a buffer (not a parameter)
        position = torch.arange(max_frames).unsqueeze(1)            # (max_frames, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )                                                            # (d_model // 2,)

        pe = torch.zeros(max_frames, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                # odd dims

        self.register_buffer('pe', pe)                              # (max_frames, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_frames, d_model)
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)