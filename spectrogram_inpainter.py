class SpectrogramInpainter(nn.Module):
    """
    Masked autoencoder-style transformer for log-magnitude spectrogram inpainting.

    At masked positions, the input frame is replaced with a learned mask embedding
    before being projected and passed through the transformer. The model predicts
    log-magnitude values for every position, but loss is only computed on masked ones.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Learned embedding that replaces masked frames in the input
        self.mask_embedding = nn.Parameter(torch.randn(config.n_bins))

        # Project each frame (n_bins,) into the transformer's working dimension
        self.input_projection = nn.Linear(config.n_bins, config.d_model)

        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_frames, config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,     # (batch, seq, features) convention throughout
            norm_first=True,      # pre-norm (more stable training than post-norm)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Project transformer output back to frequency bin space
        self.output_projection = nn.Linear(config.d_model, config.n_bins)

    def forward(
        self,
        frames: torch.Tensor,       # (batch, n_frames, n_bins)  log-magnitude
        mask: torch.Tensor,         # (batch, n_frames)           bool, True = masked/clipped
    ) -> torch.Tensor:
        """
        Returns predicted log-magnitude for all frames, shape (batch, n_frames, n_bins).
        At inference you only use the predictions at mask==True positions.
        """
        # Replace masked frame values with the learned mask embedding
        # mask.unsqueeze(-1) broadcasts to (batch, n_frames, n_bins)
        x = torch.where(mask.unsqueeze(-1), self.mask_embedding, frames)

        # Project to d_model and add positional encoding
        x = self.input_projection(x)    # (batch, n_frames, d_model)
        x = self.positional_encoding(x)

        # Bidirectional self-attention — every frame attends to every other frame
        x = self.transformer(x)         # (batch, n_frames, d_model)

        # Project back to spectrogram space
        return self.output_projection(x)  # (batch, n_frames, n_bins)