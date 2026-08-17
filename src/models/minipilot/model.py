import torch
import torch.nn as nn

from models.driving_model import DrivingModel
from models.minipilot.encoder.image_encoder import VisionEncoder
from models.minipilot.encoder.state_encoder import StateEncoder
from models.minipilot.policy import DrivingPolicy
from models.minipilot.world import TransitionModel
from mp_config import Config


class MiniPilot(DrivingModel):
    def __init__(self, config: Config):
        super(MiniPilot, self).__init__(config=config)

        token_dim = config.model.dim_world_states
        num_states = config.model.num_world_states

        self.img_enc = VisionEncoder(
            timm_model_cfg=dict(
                model_name="fastvit_t8",
                pretrained=False,
            ),
            stage_proj_dims=(None, None, None, token_dim),
        )

        self.cloud_encoder = StateEncoder(
            input_dim=3,
            token_dim=token_dim,
        )

        self.egoveh_encoder = StateEncoder(
            input_dim=19,
            token_dim=token_dim,
        )

        self.intent_encoder = StateEncoder(
            input_dim=3,
            token_dim=token_dim,
        )

        self.world_tokens = nn.Parameter(torch.randn(num_states, token_dim))

        self.world_model = TransitionModel(
            token_dim=token_dim,
            num_heads=4,
            ffn_scale=4,
            activation=nn.GELU,
            dropout=0.1,
        )

        self.policy = DrivingPolicy(
            token_dim=token_dim,
            future_steps=config.model.num_future_steps,
        )

    def prior_tokens(self, batch_size: int) -> torch.Tensor:
        return self.world_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        images: torch.Tensor,
        clouds: torch.Tensor,
        ego_state: torch.Tensor,
        ego_intent: torch.Tensor,
        updated_world_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape [Batch, Channels, Height, Width]
            clouds: Tensor of shape [Batch, Num_Points, Point_Dim]
            ego_state: Tensor of shape [Batch, State_Dim]
            ego_intent: Tensor of shape [Batch, Intent_Dim]
            world_tokens: Tensor of shape [Batch, Num_World_States, Token_Dim] (optional)
        Returns:
            action: Tensor of shape [Batch, 3]
        """

        batch_size = images.size(0)
        drive_intent_token = self.intent_encoder(ego_intent)
        vehicle_state_token = self.egoveh_encoder(ego_state)
        point_cloud_tokens = self.cloud_encoder(clouds)
        camera_image_tokens = self.img_enc(images)[-1]  # Get the last stage features
        camera_image_tokens = camera_image_tokens.flatten(2, 3).permute(0, 2, 1)

        if updated_world_tokens is None:
            updated_world_tokens = self.prior_tokens(batch_size)

        current_context_tokens = torch.cat(
            [
                camera_image_tokens,
                point_cloud_tokens,
                vehicle_state_token,
                drive_intent_token,
            ],
            dim=-2,
        )

        updated_world_tokens, future_context_tokens = self.world_model(
            current_context_tokens,
            updated_world_tokens,
        )

        trajectory = self.policy(updated_world_tokens)

        return dict(
            future_trajectory=trajectory,
            updated_world_tokens=updated_world_tokens,
            current_context_tokens=current_context_tokens,
            future_context_tokens=future_context_tokens,
        )

    def _shared_step(self, mode: str, batch, batch_idx: int) -> torch.Tensor:
        stamps = batch["stamps"]
        states = batch["states"]
        images = batch["images"]
        clouds = batch["clouds"]
        intents = batch["intents"]
        labels = batch["labels"]

        batch_size = stamps.size(0)
        seq_length = stamps.size(1)

        # Original code assumed sequence-first iteration; keep the same
        # permutation logic so time becomes the leading dimension to iterate.
        stamps = stamps.transpose(0, 1)  # [SEQ, BATCH, ...]
        states = states.transpose(0, 1)  # [SEQ, BATCH, ...]
        images = images.transpose(0, 1)  # [SEQ, BATCH, ...]
        clouds = clouds.transpose(0, 1)  # [SEQ, BATCH, ...]
        labels = labels.transpose(0, 1)  # [SEQ, BATCH, ...]
        intents = intents.transpose(0, 1)  # [SEQ, BATCH, ...]

        estimated_context_tokens = None
        current_world_tokens = self.prior_tokens(batch_size=batch_size)

        step_losses = []
        timeseries = zip(stamps, states, images, clouds, intents, labels)

        for idx, (stamp, state, image, cloud, intent, label) in enumerate(timeseries):
            result_dict = self(image, cloud, state, intent, current_world_tokens)

            traj_loss = sense_loss = None

            current_context_tokens = result_dict["current_context_tokens"]
            estimated_trajectory = result_dict["future_trajectory"]
            recorded_trajectory = label

            traj_loss = self.criterion(
                estimated_trajectory,
                recorded_trajectory,
            )

            if (
                current_context_tokens is not None
                and estimated_context_tokens is not None
            ):
                sense_loss = self.criterion(
                    current_context_tokens, estimated_context_tokens
                )

            current_world_tokens = result_dict["updated_world_tokens"]
            estimated_context_tokens = result_dict["future_context_tokens"]

            if traj_loss and sense_loss:

                step_loss = (
                    self.config.model.traj_bias * traj_loss
                    + self.config.model.sense_bias * sense_loss
                )

                step_losses.append(step_loss)

        normloss = torch.stack(step_losses).mean()

        self.log(
            f"{mode}_loss",
            normloss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return normloss  # Average loss over the sequence

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step("train", batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step("val", batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dloader_idx: int = 0) -> torch.Tensor:
        stamps, states, images, clouds, intents, labels = tuple(batch.values())
        return self(
            images, clouds, states, intents, None
        )  # No prior world tokens for prediction
