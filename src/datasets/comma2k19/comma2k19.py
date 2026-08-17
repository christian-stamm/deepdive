from pathlib import Path

import numpy as np
import torch
from matplotlib import image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Resize, ToTensor, transforms

from .utils.framereader import FrameReader
from .utils.mathops import (
    consolidate_series,
    ecef_to_ned,
    ecef_to_traj,
    interpolate_series,
)


class CommaTransform(transforms.Compose):
    def __init__(self, image_size: torch.Size = torch.Size([512, 256])):
        self.image_size = image_size
        self.image_tf = transforms.Compose(
            [
                ToTensor(),
                Resize(image_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, sample: dict) -> dict:
        sample["images"] = self.image_tf(sample["images"])
        return sample


class CommaDataset(Dataset):

    def __init__(
        self, root: Path, future_steps: int = 5, transform=CommaTransform()
    ) -> None:
        self.root = root
        self.transform = transform
        self.future_steps = future_steps

        self.frame_reader = FrameReader(str(root / "video.hevc"))
        self.stamps = self._load_glob_pose(root, "frame_times")

        ecef_pose = self._load_glob_pose(root, "frame_positions")
        ecef_speed = self._load_glob_pose(root, "frame_velocities")
        quat_orient = self._load_glob_pose(root, "frame_orientations")

        ego_pos = ecef_to_traj(ecef_pose, quat_orient)
        ego_vel = ecef_to_ned(ecef_speed, quat_orient)
        ego_accel = self._load_proc_log(root, "IMU", "accelerometer")
        ego_gyro = self._load_proc_log(root, "IMU", "gyro")

        ego_speed = self._load_proc_log(root, "CAN", "speed")
        radar_dist = self._load_proc_log(root, "CAN", "radar")
        steer_angle = self._load_proc_log(root, "CAN", "steering_angle")

        valid_radar_cols = [
            col for col in range(radar_dist.shape[-1]) if col not in (3, 4)
        ]

        radar_dist = radar_dist[:, valid_radar_cols]
        self.states = torch.cat(
            [
                ego_pos,
                ego_vel,
                ego_accel,
                ego_gyro,
                ego_speed,
                radar_dist,
                steer_angle,
            ],
            dim=-1,
        )

    def _load_series(self, file: Path) -> torch.Tensor:
        if not file.exists():
            raise FileNotFoundError("Could not load Comma Series File: ", str(file))

        return torch.from_numpy(np.load(file)).float()

    def _load_glob_pose(self, root: Path, file: str) -> torch.Tensor:
        return self._load_series(root / "global_pose" / file)

    def _load_proc_log(self, root: Path, sensor: str, group: str) -> torch.Tensor:
        basedir = root / "processed_log" / sensor / group
        values = self._load_series(basedir / "value")
        stamps = self._load_series(basedir / "t")

        if stamps.shape[0] != values.shape[0]:
            raise ValueError(
                f"Mismatch in number of timestamps and values for {sensor}/{group}: "
                f"{stamps.shape[0]} timestamps vs {values.shape[0]} values"
            )

        if values.ndim == 1:
            values = values.unsqueeze(-1)

        stamps, values = consolidate_series(stamps, values)
        return interpolate_series(self.stamps.flatten(), stamps.flatten(), values)

    def _load_frame(self, idx: int) -> torch.Tensor:
        frame = self.frame_reader.get(idx, 1, pix_fmt="rgb24").pop(0)
        frame = np.array(frame)
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1) / 255.0
        return frame

    def _load_label(self, idx: int) -> torch.Tensor:
        lower_bound = idx + 1
        upper_bound = lower_bound + self.future_steps

        if self.stamps.size(0) < upper_bound:
            raise IndexError(
                f"Index {idx} with future_steps {self.future_steps} (upperbound={upper_bound}) exceeds dataset length {len(self)}"
            )

        traj = self.states[lower_bound:upper_bound, 0:2]
        return traj

    def __getitem__(self, idx: int):
        sample = {
            "stamps": self.stamps[idx],
            "states": self.states[idx],
            "images": self._load_frame(idx),
            "clouds": torch.zeros(1, 3),  # Placeholder for point cloud data
            "intents": torch.zeros(3),  # Placeholder for intent data
            "labels": self._load_label(idx),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return max(0, self.stamps.size(0) - self.future_steps - 1)
