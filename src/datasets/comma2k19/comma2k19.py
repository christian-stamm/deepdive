from pathlib import Path
import pprint
from typing import Callable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Resize, ToTensor, transforms

from .utils.camera import device_from_ecef
from .utils.framereader import FrameReader


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
        sample["sensors"]["camera_imgs"] = self.image_tf(
            sample["sensors"]["camera_imgs"]
        )
        return sample


class Sensor:

    def __init__(self, name: str, stamps: np.ndarray, values: np.ndarray):
        self.name = name
        self.stamps = stamps
        self.values = values

    @staticmethod
    def load(path: Path):
        data = np.load(path, mmap_mode="r")
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data

    @classmethod
    def from_file(cls, path: Path):
        stamps = Sensor.load(path / "t")
        values = Sensor.load(path / "value")
        return cls(path.stem, stamps, values)

    def __getitem__(self, idx: int | slice | np.ndarray):
        stamps = self.stamps[idx]
        values = self.values[idx]
        return np.concatenate([stamps, values], axis=-1)


class RecordInfo:

    def __init__(self, path: Path):
        if not path.is_dir():
            path = path.parent

        self.path = path
        self.stamps = Sensor.load(path / "global_pose/frame_times")

    def __len__(self):
        return len(self.stamps)

    @property
    def chunk(self):
        return int(self.path.parts[-3].split("_")[-1])

    @property
    def route(self):
        return self.path.parts[-2]

    @property
    def dongle(self):
        return self.route.split("|")[0]

    @property
    def stamp(self):
        return self.route.split("|")[1]

    @property
    def segment(self):
        return int(self.path.parts[-1])


class Recording:

    def __init__(self, info: RecordInfo):
        self.info = info

        reader = FrameReader(str(info.path / "video.hevc"))
        images = reader.get(0, len(info), pix_fmt="rgb24")
        self.camera = Sensor("camera", info.stamps, np.stack(images))
        del reader, images

        pos = Sensor.load(info.path / "global_pose/frame_positions")
        vel = Sensor.load(info.path / "global_pose/frame_velocities")
        orient = Sensor.load(info.path / "global_pose/frame_orientations")
        self.positions = Sensor("position", info.stamps, pos)
        self.velocities = Sensor("velocity", info.stamps, vel)
        self.orientations = Sensor("orientation", info.stamps, orient)
        del pos, vel, orient

        self.accel = Sensor.from_file(info.path / "processed_log/IMU/accelerometer")
        self.gyro = Sensor.from_file(info.path / "processed_log/IMU/gyro")
        self.speed = Sensor.from_file(info.path / "processed_log/CAN/speed")
        self.radar = Sensor.from_file(info.path / "processed_log/CAN/radar")
        self.steer = Sensor.from_file(info.path / "processed_log/CAN/steering_angle")


class CommaDataset(Dataset):

    def __init__(
        self,
        root: Path,
        sample_rate: float = 20,
        future_steps: int = 60,
        transform: Callable[[dict], dict] | None = CommaTransform(),
    ) -> None:
        self.root = root
        self.transform = transform
        self.sample_rate = sample_rate
        self.future_steps = future_steps
        self.records = [RecordInfo(path) for path in self.root.rglob("**/video.hevc")]

    def __getitem__(self, idx: int):
        recinfo = self.records[idx]

        anchors = np.arange(len(recinfo) - self.future_steps)
        horizon = np.arange(self.future_steps)
        windows = anchors[:, None] + horizon[None, :]
        windows = windows.astype(np.float32)
        windows *= 1.0 / self.sample_rate

        frames = recinfo.stamps.squeeze(-1)
        indices = np.searchsorted(frames, windows.ravel(), side="right")
        indices = indices.reshape(windows.shape) - 1

        print("indices: ", indices.shape, indices[0, :10])

        # recording = Recording(recinfo)
        # positions = recording.positions[windows]
        # velocities = recording.velocities[windows]
        # orientations = recording.orientations[windows]

        return windows

    def __len__(self):
        return len(self.records)
