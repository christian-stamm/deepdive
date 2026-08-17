import torch
import torch.nn as nn


class DecisionBlock(nn.Module):
    def __init__(
        self,
        future_steps: int = 5,
        token_dim: int = 64,
        num_heads: int = 4,
        ffn_scale: int = 4,
        activation: nn.Module = nn.GELU,
        batch_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps

        self.mha = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_scale * token_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ffn_scale * token_dim, 2),
        )

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:

        query = self.norm1(query)

        delta, _ = self.mha(
            query=query,
            key=key,
            value=value,
        )

        result = query + delta
        result = self.norm2(result)

        return self.ffn(result)  # [Batch, Future_Steps, 2]


class DrivingPolicy(nn.Module):
    def __init__(
        self,
        token_dim: int = 64,
        future_steps: int = 5,
    ):
        super().__init__()

        self.prior_token = nn.Parameter(torch.randn(1, future_steps, token_dim))

        self.planner = DecisionBlock(
            future_steps=future_steps,
            token_dim=token_dim,
            num_heads=4,
            ffn_scale=4,
            activation=nn.GELU,
            batch_first=True,
        )

    def forward(self, world_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            world_tokens: Tensor of shape [Batch, Num_Tokens, Feature_Dim]
            intent_token: Tensor of shape [Batch, 1, Feature_Dim]
        Returns:
            action: Tensor of shape [Batch, Action_Dim]
        """

        query_token = self.prior_token.expand(
            world_tokens.size(0), -1, -1
        )  # [Batch, 1,Token_Dim]

        trajectory = self.planner(
            query=query_token,
            key=world_tokens,
            value=world_tokens,
        )

        return trajectory  # for Batched (x, y) coordinates of n Future Steps
