import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class VisionPreprocessor(nn.Module):
    def __init__(
        self,
        size: tuple = (512, 256),
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super(VisionPreprocessor, self).__init__()
        self.rescaler = transforms.Resize(size)
        self.normalizer = transforms.Normalize(mean=mean, std=std)

    @staticmethod
    def batchify(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(
                f"Expected a 3D/4D tensor [B, C, H, W], got shape {tuple(x.shape)}"
            )

        return x

    @staticmethod
    def tensorfy(x: torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(x, Image.Image):
            x = TF.to_tensor(x)

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor or PIL.Image.Image, got {type(x)}"
            )

        return x

    def forward(self, x: torch.Tensor | Image.Image) -> torch.Tensor:
        x = self.tensorfy(x)
        x = self.batchify(x)
        x = self.rescaler(x)
        x = self.normalizer(x)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        timm_model_cfg: dict = dict(
            model_name="fastvit_t12",
            pretrained=False,
            transforms=VisionPreprocessor(),
        ),
        stage_proj_dims: tuple = (None, None, None, 128),
    ):
        super().__init__()

        timm_model_cfg.pop("fork_feat", None)
        tf = timm_model_cfg.pop("transforms", None)
        self.preprocessor = tf if tf is not None else VisionPreprocessor()

        self.encoder = timm.models.create_model(**timm_model_cfg, fork_feat=True)

        sample = torch.randn(
            1, 3, 32, 32
        )  # Sample input to determine feature dimensions

        features = self.encoder(self.preprocessor(sample))
        self.projections = nn.ModuleList()

        for stage, (tokens, proj_dim) in enumerate(zip(features, stage_proj_dims)):
            if proj_dim and 0 < proj_dim:
                self.projections.append(
                    nn.Conv2d(tokens.shape[1], proj_dim, kernel_size=1)
                )
            else:
                self.projections.append(nn.Identity())

    def forward(self, x: torch.Tensor | Image.Image) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, Channels, Height, Width] or PIL Image
        Returns:
            features: Tensor of shape [Batch, Feature_Dim]
        """
        output = dict()

        preprocessed = self.preprocessor(x)
        encoder_param = next(self.encoder.parameters())
        preprocessed = preprocessed.to(
            device=encoder_param.device,
            dtype=encoder_param.dtype,
        )
        features = self.encoder(preprocessed)

        output = {}
        for stage, (tokens, project) in enumerate(zip(features, self.projections)):
            projected = project(tokens)
            if not isinstance(project, nn.Identity):
                output[f"stage_{stage:02d}"] = projected

        return list(output.values()) if 0 < len(output) else None
