from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils.camera import device_from_ecef
from .utils.framereader import FrameReader

PADDED_LENGTH = 1200


def pad_sequence(x: np.ndarray, length: int, value: float = np.nan):
    pad_width = [(0, length - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad_width, mode="constant", constant_values=value)


class Sensor:

    def __init__(self, name: str, stamps: np.ndarray, values: np.ndarray):
        self.name = name
        self.stamps = pad_sequence(stamps, length=PADDED_LENGTH)
        self.values = pad_sequence(values, length=PADDED_LENGTH)

    @staticmethod
    def batchify(name: str, sensors: list["Sensor"]):
        valid = all(sensor.name == name for sensor in sensors)

        if not valid:
            raise ValueError("All sensors must have the same name to batchify.")

        return Sensor(
            name,
            np.stack([sensor.stamps for sensor in sensors]),
            np.stack([sensor.values for sensor in sensors]),
        )

    def interp(self, stamps: np.ndarray):
        upper = np.searchsorted(self.stamps, stamps, side="left")
        upper = np.clip(upper, 1, len(self.stamps) - 1)
        lower = upper - 1

        t0 = self.stamps[lower]
        v0 = self.values[lower]

        t1 = self.stamps[upper]
        v1 = self.values[upper]

        alpha = (stamps - t0) / (t1 - t0)
        alpha = alpha[..., None]

        return v0 + alpha * (v1 - v0)

    def nearest(self, stamps: np.ndarray):
        fltstamps = stamps.ravel()
        indices = np.searchsorted(self.stamps, fltstamps, side="right") - 1

        valid = indices >= 0
        indices = np.clip(indices, 0, len(self) - 1)
        valid = valid.reshape(stamps.shape)

        return np.where(valid[..., None], self.values[indices], np.nan)

    @staticmethod
    def load(path: Path):
        return np.load(path, mmap_mode="r")

    @classmethod
    def from_files(cls, path: Path):
        stamps = Sensor.load(path / "t")
        values = Sensor.load(path / "value")

        if values.ndim == 1:
            values = values[:, None]

        return cls(path.stem, stamps, values)

    def __getitem__(self, idx: int | slice | np.ndarray):
        stamps = self.stamps[idx]
        values = self.values[idx]
        return Sensor(self.name, stamps, values)

    def __len__(self):
        return self.stamps.shape[0]


class RecordMeta:

    def __init__(self, path: Path):
        if not path.is_dir():
            path = path.parent

        self.path = path

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

    def __init__(
        self,
        meta: RecordMeta,
        images: Sensor,
        pose: Sensor,
        accel: Sensor,
        gyro: Sensor,
        speed: Sensor,
        radar: Sensor,
        steer: Sensor,
    ):
        self.meta = meta
        self.camera = images
        self.pose = pose
        self.accel = accel
        self.gyro = gyro
        self.speed = speed
        self.radar = radar
        self.steer = steer

    def __len__(self):
        return len(self.camera)

    @classmethod
    def from_drive(cls, root: Path):
        meta = RecordMeta(root)
        stamps = Sensor.load(meta.path / "global_pose/frame_timestamps")

        reader = FrameReader(str(meta.path / "video.hevc"))
        images = reader.get(0, len(stamps), pix_fmt="rgb24")
        images = np.stack(images)
        positions = Sensor.load(meta.path / "global_pose/frame_positions")
        velocities = Sensor.load(meta.path / "global_pose/frame_velocities")
        orientations = Sensor.load(meta.path / "global_pose/frame_orientations")
        poses = np.concatenate([positions, velocities, orientations], axis=-1)

        cls(
            meta=meta,
            images=Sensor("camera", stamps, images),
            pose=Sensor("pose", stamps, poses),
            accel=Sensor.from_files(meta.path / "processed_log/IMU/accelerometer"),
            gyro=Sensor.from_files(meta.path / "processed_log/IMU/gyro"),
            speed=Sensor.from_files(meta.path / "processed_log/CAN/speed"),
            radar=Sensor.from_files(meta.path / "processed_log/CAN/radar"),
            steer=Sensor.from_files(meta.path / "processed_log/CAN/steering_angle"),
        )

    def get_chunks(self, history: float, horizon: float, rate: float):
        dt = 1.0 / rate
        t0 = self.info.stamps[0]
        tN = self.info.stamps[-1]

        anchors = np.arange(t0 + history, tN - horizon, dt)
        trange = np.arange(-history, horizon, dt)

        windows = anchors[:, None] + trange[None, :]
        windows = windows.astype(np.float32)
        split = int(history / dt)
        return windows, split

    def get_input(self, stamps: np.ndarray):
        accel = self.accel.collect_causal(stamps)
        gyro = self.gyro.collect_causal(stamps)
        speed = self.speed.collect_causal(stamps)
        radar = self.radar.collect_causal(stamps)
        steer = self.steer.collect_causal(stamps)
        camera = self.camera.collect_causal(stamps)

        return {
            "accel": torch.from_numpy(accel),
            "gyro": torch.from_numpy(gyro),
            "speed": torch.from_numpy(speed),
            "radar": torch.from_numpy(radar),
            "steer": torch.from_numpy(steer),
            "camera": torch.from_numpy(camera),
        }

    def get_label(self, stamps: np.ndarray):
        pos = self.positions.collect_interp(stamps)
        speed = self.speed.collect_interp(stamps)
        steer = self.steer.collect_interp(stamps)

        return {
            "position": torch.from_numpy(pos),
            "speed": torch.from_numpy(speed),
            "steer": torch.from_numpy(steer),
        }


class CommaDataset(Dataset):

    def __init__(
        self,
        root: Path,
        rate: float = 20,
        history: float = 0.5,
        horizon: float = 3.0,
    ) -> None:
        self.root = root
        self.rate = rate
        self.history = history
        self.horizon = horizon
        self.records = [path.parent for path in self.root.rglob("**/video.hevc")]

    def __getitem__(self, idx: int):
        return Recording.from_drive(self.records[idx])

    @staticmethod
    def collate_fn(batch: list[Recording]) -> Recording:
        return Recording(
            meta=[record.meta for record in batch],
            images=Sensor.batchify("camera", [record.camera for record in batch]),
            pose=Sensor.batchify("pose", [record.pose for record in batch]),
            accel=Sensor.batchify("accel", [record.accel for record in batch]),
            gyro=Sensor.batchify("gyro", [record.gyro for record in batch]),
            speed=Sensor.batchify("speed", [record.speed for record in batch]),
            radar=Sensor.batchify("radar", [record.radar for record in batch]),
            steer=Sensor.batchify("steer", [record.steer for record in batch]),
        )

    def __len__(self):
        return len(self.records)
