from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Resize, ToTensor, transforms

from .utils.camera import device_from_ecef

from .utils.framereader import FrameReader
from .utils.timeseries import (
    consolidate_series,
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

        self.ecef_pos = self._load_glob_pose(root, "frame_positions").numpy()
        self.ecef_vel = self._load_glob_pose(root, "frame_velocities").numpy()
        self.ecef_rot = self._load_glob_pose(root, "frame_orientations").numpy()

        self.ego_accel = self._load_proc_log(root, "IMU", "accelerometer")
        self.ego_gyro = self._load_proc_log(root, "IMU", "gyro")

        self.ego_speed = self._load_proc_log(root, "CAN", "speed")
        self.radar_dist = self._load_radar_filtered(root, "CAN", "radar")
        self.steer_angle = self._load_proc_log(root, "CAN", "steering_angle")

       

    def _load_series(self, file: Path) -> torch.Tensor:
        if not file.exists():
            raise FileNotFoundError("Could not load Comma Series File: ", str(file))

        return torch.from_numpy(np.load(file)).float()

    def _load_glob_pose(self, root: Path, file: str) -> torch.Tensor:
        return self._load_series(root / "global_pose" / file)

    def _load_radar_filtered(self, root: Path, bus:str, group:str) -> torch.Tensor:
        radar_dist = self._load_proc_log(root, bus, group)
        radar_nan_mask = np.isnan(radar_dist).any(dim=0).to(torch.bool)

        if radar_nan_mask.any():
            print("Warning: NaN values found in radar distance data. These will be filtered out.")

        return radar_dist[:, ~radar_nan_mask]

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

    def _fetch_labels(self, idx: int) -> torch.Tensor:
        lower_bound = idx + 1
        upper_bound = lower_bound + self.future_steps

        if self.stamps.size(0) < upper_bound:
            raise IndexError(
                f"Index {idx} with future_steps {self.future_steps} (upperbound={upper_bound}) exceeds dataset length {len(self)}"
            )

        return device_from_ecef(self.ecef_pos[idx], self.ecef_rot[idx], self.ecef_pos[lower_bound:upper_bound])

    def _fetch_sensors(self, idx: int) -> dict:
       state = {
            "stamps": self.stamps[idx],
            "ecef_pos": self.ecef_pos[idx],
            "ecef_vel": self.ecef_vel[idx],
            "ecef_rot": self.ecef_rot[idx],
            "ego_accel": self.ego_accel[idx],
            "ego_gyro": self.ego_gyro[idx],
            "ego_speed": self.ego_speed[idx],
            "camera_imgs": self._load_frame(idx),
            "radar_dist": self.radar_dist[idx],
            "radar_clouds": torch.zeros(1, 3),  # Placeholder for radar point cloud data
            "steer_angle": self.steer_angle[idx],
       }

       return state

    def __getitem__(self, idx: int):
        sample = {
            "intents": torch.zeros(1, 3),  # Placeholder for intent data
            "sensors": self._fetch_sensors(idx),
            "labels": self._fetch_labels(idx),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return max(0, self.stamps.size(0) - self.future_steps - 1)
