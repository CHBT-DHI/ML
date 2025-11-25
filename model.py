import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, n_heads=4, n_layers=2, seq_len=5, dropout=0.1):
        """
        Args:
            input_dim (int): Number of features per timestep
            model_dim (int): Dimension of transformer embeddings
            n_heads (int): Number of attention heads
            n_layers (int): Number of decoder layers
            seq_len (int): Length of input sequence
        """
        super().__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))

        # Transformer decoder layer (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Final linear layer to predict scalar
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size,) predicted NH4
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Causal mask: prevent attending to future timesteps
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # Decoder-only forward (using input as memory and tgt)
        out = self.decoder(tgt=x, memory=x, tgt_mask=mask)

        # Take the last timestep embedding for prediction
        last_timestep = out[:, -1, :]  # (batch_size, model_dim)
        scalar_pred = self.output_layer(last_timestep).squeeze(-1)  # (batch_size,)
        return scalar_pred
