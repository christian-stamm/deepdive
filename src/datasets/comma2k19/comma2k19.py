from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.camera import device_from_ecef
from .utils.framereader import FrameReader


class Sensor:

    def __init__(self, name: str, stamps: np.ndarray, values: np.ndarray):
        self.name = name
        self.stamps = stamps
        self.values = values

    def collect_interp(self, stamps: np.ndarray):
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

    def collect_causal(self, stamps: np.ndarray):
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
    def from_file(cls, path: Path):
        stamps = Sensor.load(path / "t")
        values = Sensor.load(path / "value")

        if values.ndim == 1:
            values = values[:, None]

        return cls(path.stem, stamps, values)

    def __getitem__(self, idx: int | slice | np.ndarray):
        stamps = self.stamps[idx]
        values = self.values[idx]
        return np.concatenate([stamps, values], axis=-1)

    def __len__(self):
        return self.stamps.shape[0]


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

    def get_chunks(self, history: float, horizon: float, rate: float):
        dt = 1.0 / rate
        t0 = self.info.stamps[0]
        tN = self.info.stamps[-1]

        anchors = np.arange(t0 + history, tN - horizon, dt)
        trange = np.arange(-history, horizon, dt)

        windows = anchors[:, None] + trange[None, :]
        windows = windows.astype(np.float32)

        split = int(history / dt)
        context = windows[:, :split]
        future = windows[:, split:]

        return context, future

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
        self.records = [RecordInfo(path) for path in self.root.rglob("**/video.hevc")]

    def __getitem__(self, idx: int):
        recording = Recording(self.records[idx])

        for sample in recording.sample(self.history, self.horizon, self.rate):
            yield sample

    def __len__(self):
        return len(self.records)
