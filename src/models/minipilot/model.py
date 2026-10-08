import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driving_model import DrivingModel
from models.minipilot.encoder.image_encoder import VisionEncoder
from models.minipilot.encoder.state_encoder import StateEncoder
from models.minipilot.policy import DrivingPolicy, FutureSensorHead
from models.minipilot.world import TransitionModel
from mp_config import Config


class MiniPilot(DrivingModel):
    def __init__(self, config: Config):
        super(MiniPilot, self).__init__(config=config)

        token_dim = config.model.dim_world_states
        num_states = config.model.num_world_states

        self.image_enc = VisionEncoder(
            model_cfg=dict(
                model_name="fastvit_t12.apple_dist_in1k",
                pretrained=True,
                pretrained_cfg_overlay={
                    "file": "/workspace/res/fastvit_t12.safetensors",
                },
                features_only=True,
                num_classes=0,
            ),
            stage_proj_dims=(None, None, None, token_dim),
        )

        self.cloud_encoder = StateEncoder(
            input_dim=3,
            token_dim=token_dim,
        )

        self.telemetry_enc = StateEncoder(
            input_dim=config.model.telemetry_dim,
            token_dim=token_dim,
        )

        self.intent_encoder = StateEncoder(
            input_dim=3,
            token_dim=token_dim,
        )
        self.dt_encoder = StateEncoder(
            input_dim=1,
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
        self.future_sensor_head = FutureSensorHead(
            token_dim=token_dim,
            future_steps=config.model.num_future_steps,
            output_dim=config.model.telemetry_dim,
        )

    def prior_tokens(self, batch_size: int) -> torch.Tensor:
        return self.world_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def train(self, mode: bool = True) -> None:
        super(MiniPilot, self).train(mode)
        self.image_enc.freeze(freezed=True)

    def forward(
        self,
        images: torch.Tensor,
        clouds: torch.Tensor,
        telemetry: torch.Tensor,
        intents: torch.Tensor,
        world_tokens: torch.Tensor = None,
        delta_time: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape [Batch, Channels, Height, Width]
            clouds: Tensor of shape [Batch, Num_Points, Point_Dim]
            telemetry: Tensor of shape [Batch, Telemetry_Dim]
            intents: Tensor of shape [Batch, Intent_Dim]
            world_tokens: Tensor of shape [Batch, Num_World_States, Token_Dim] (optional)
        Returns:
            action: Tensor of shape [Batch, 3]
        """

        batch_size = images.size(0)
        drive_intent_token = self.intent_encoder(intents)
        telemetry_tokens = self.telemetry_enc(telemetry)
        pointcloud_tokens = self.cloud_encoder(clouds)
        if delta_time is None:
            delta_time = telemetry.new_zeros(batch_size, 1)
        elif delta_time.dim() == 1:
            delta_time = delta_time.unsqueeze(-1)
        dt_token = self.dt_encoder(delta_time)
        cameraimage_tokens = self.image_enc(images)[-1]  # Get the last stage features
        cameraimage_tokens = cameraimage_tokens.flatten(2, 3).permute(0, 2, 1)

        if world_tokens is None:
            world_tokens = self.prior_tokens(batch_size)

        current_context_tokens = torch.cat(
            [
                cameraimage_tokens,
                pointcloud_tokens,
                telemetry_tokens,
                drive_intent_token,
            ],
            dim=-2,
        )

        world_tokens, future_context_tokens = self.world_model(
            current_context_tokens,
            world_tokens,
            delta_time=dt_token,
        )

        trajectory = self.policy(world_tokens, drive_intent_token)
        future_sensor = self.future_sensor_head(world_tokens)

        return dict(
            future_trajectory=trajectory,
            updated_world_tokens=world_tokens,
            current_context_tokens=current_context_tokens,
            future_context_tokens=future_context_tokens,
            future_sensor=future_sensor,
        )

    def _shared_step(self, mode: str, batch, batch_idx: int) -> torch.Tensor:
        sensors, intents, labels, future_telemetry, future_telemetry_valid = self._unpack_batch(batch)
        stamps, images, clouds, telemetry = sensors

        batch_size = stamps.size(0)
        seq_length = stamps.size(1)

        # Original code assumed sequence-first iteration; keep the same
        # permutation logic so time becomes the leading dimension to iterate.
        stamps = stamps.transpose(0, 1)  # [SEQ, BATCH, ...]
        telemetry = telemetry.transpose(0, 1)  # [SEQ, BATCH, ...]
        images = images.transpose(0, 1)  # [SEQ, BATCH, ...]
        clouds = clouds.transpose(0, 1)  # [SEQ, BATCH, ...]
        labels = labels.transpose(0, 1)  # [SEQ, BATCH, ...]
        intents = intents.transpose(0, 1)  # [SEQ, BATCH, ...]
        future_telemetry = future_telemetry.transpose(0, 1)
        future_telemetry_valid = future_telemetry_valid.transpose(0, 1)

        current_world_tokens = self.prior_tokens(batch_size=batch_size)

        step_losses = []
        previous_stamps = None
        timeseries = zip(
            stamps,
            images,
            clouds,
            telemetry,
            intents,
            labels,
            future_telemetry,
            future_telemetry_valid,
        )

        for idx, (stamp, image, cloud, state, intent, label, sensor_target, sensor_valid) in enumerate(timeseries):
            if previous_stamps is None:
                delta_time = stamp.new_zeros(stamp.shape)
            else:
                delta_time = stamp - previous_stamps
            result_dict = self(
                image,
                cloud,
                state,
                intent,
                current_world_tokens,
                delta_time,
            )

            current_context_tokens = result_dict["current_context_tokens"]
            estimated_trajectory = result_dict["future_trajectory"]
            recorded_trajectory = label

            trajectory_error = F.smooth_l1_loss(
                estimated_trajectory,
                recorded_trajectory[:, :, :2],  # Only consider x, y for trajectory loss
                reduction="none",
            )
            horizon_weight = torch.linspace(
                1.0,
                0.5,
                trajectory_error.size(1),
                device=trajectory_error.device,
            ).view(1, -1, 1)
            traj_loss = (trajectory_error * horizon_weight).mean()

            current_world_tokens = result_dict["updated_world_tokens"]
            sensor_prediction = result_dict["future_sensor"]
            sensor_error = F.smooth_l1_loss(
                sensor_prediction,
                sensor_target,
                reduction="none",
            )
            sensor_mask = sensor_valid.to(sensor_error.dtype)
            sensor_loss = (sensor_error * sensor_mask).sum() / sensor_mask.sum().clamp_min(1.0)

            step_loss = (
                self.config.model.traj_bias * traj_loss
                + self.config.model.sensor_bias * sensor_loss
            )
            step_losses.append(step_loss)
            previous_stamps = stamp

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
        sensors, intents, labels, _, _ = self._unpack_batch(batch)
        stamps, images, clouds, telemetry = sensors
        batch_size, sequence_length = stamps.shape[:2]
        world_tokens = self.prior_tokens(batch_size)
        outputs = []
        previous_stamps = None
        for step in range(sequence_length):
            stamp = stamps[:, step]
            delta_time = stamp.new_zeros(stamp.shape) if previous_stamps is None else stamp - previous_stamps
            result = self(
                images[:, step],
                clouds[:, step],
                telemetry[:, step],
                intents[:, step],
                world_tokens,
                delta_time,
            )
            outputs.append(result["future_trajectory"])
            world_tokens = result["updated_world_tokens"].detach()
            previous_stamps = stamp
        return torch.stack(outputs, dim=1)

    def unpack_sensors(self, sensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stamps = sensors["stamps"]
        images = sensors["camera_imgs"]
        clouds = sensors["radar_clouds"]
        telemetry = torch.cat(
            [
                sensors["ego_accel"],
                sensors["ego_gyro"],
                sensors["ego_speed"],
                sensors["radar_dist"],
                sensors["steer_angle"],
            ],
            dim=-1,
        )

        return stamps, images, clouds, telemetry

    def _unpack_batch(self, batch: dict[str, torch.Tensor]):
        sensors = self.unpack_sensors(batch["sensors"])
        return (
            sensors,
            batch["intents"],
            batch["labels"],
            batch["future_telemetry"],
            batch["future_telemetry_valid"],
        )
