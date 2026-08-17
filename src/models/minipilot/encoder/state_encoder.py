import torch
from torch import nn


class StateEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        token_dim: int = 64,
        dropout: float = 0.1,
    ):
        super(StateEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.input_dim = input_dim
        self.token_dim = token_dim

    def batchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected a 1D/2D tensor of shape [Batch, {self.input_dim}], got {tuple(x.shape)}"
            )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, Channels, Height, Width] or PIL Image
        Returns:
            features: Tensor of shape [Batch, Feature_Dim]
        """
        x = self.batchify(x)
        x = self.encoder(x)
        return x.view(-1, 1, self.token_dim)
