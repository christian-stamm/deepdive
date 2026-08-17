import torch
import torch.nn as nn


class TransitionBlock(nn.Module):
    def __init__(
        self,
        token_dim: int = 64,
        num_heads: int = 4,
        ffn_scale: int = 4,
        activation: nn.Module = nn.GELU,
        batch_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

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
            nn.Linear(ffn_scale * token_dim, token_dim),
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

        state = query + delta
        state = self.norm2(state)
        state = state + self.ffn(state)
        state = self.norm2(state)

        return state


class TransitionModel(nn.Module):

    def __init__(
        self,
        token_dim: int = 64,
        num_heads: int = 4,
        ffn_scale: int = 4,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.state_tf = TransitionBlock(
            token_dim=token_dim,
            num_heads=num_heads,
            ffn_scale=ffn_scale,
            activation=activation,
            batch_first=True,
            dropout=dropout,
        )

        self.sensor_tf = TransitionBlock(
            token_dim=token_dim,
            num_heads=num_heads,
            ffn_scale=ffn_scale,
            activation=activation,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        world_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transition model.
        Args:
            context_tokens: Contextual Tokens (B, C, D) tensor of context tokens
            world_tokens: World State (B, W, D) tensor of world state tokens
        Returns:
            next_world_state_tokens: (B, W, D) tensor of next world state token
            next_sensor_tokens: (B, S, D) tensor of next sensor tokens

        """

        drive_context = torch.cat(
            [
                context_tokens,
                world_tokens,
            ],
            dim=1,
        )

        updated_world_tokens = self.state_tf(
            query=world_tokens,
            key=drive_context,
            value=drive_context,
        )

        updated_world_tokens = self.state_tf(
            query=updated_world_tokens,
            key=updated_world_tokens,
            value=updated_world_tokens,
        )

        future_context_tokens = self.sensor_tf(
            query=context_tokens,
            key=updated_world_tokens,
            value=updated_world_tokens,
        )

        return updated_world_tokens, future_context_tokens
