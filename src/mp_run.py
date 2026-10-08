from tqdm import tqdm
import torch

from torch.utils.data import DataLoader
from datasets.comma2k19.comma2k19 import CommaDataset
from models.minipilot.model import MiniPilot
from mp_config import Config
import cv2
import numpy as np

from datasets.comma2k19.utils.camera import img_from_device, denormalize

def draw_path(
    device_path,
    img,
    color=(255, 0, 0),  # blue in BGR
    thickness=12,
    alpha=0.35,
):
    device_path += np.array([0.0, 0.0, 2.])  # offset to camera height
    
    img_pts = denormalize(img_from_device(device_path))

    valid = np.isfinite(img_pts).all(axis=1)
    img_pts = img_pts[valid]

    if len(img_pts) < 2:
        return

    pts = img_pts.astype(np.int32).reshape(-1, 1, 2)

    overlay = img.copy()

    cv2.polylines(
        overlay,
        [pts],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1.0 - alpha,
        0,
        dst=img,
    )


def load_cfg(path: str = "res/config.yaml") -> Config:
    config = Config()

    try:
        config = Config.from_yaml(path)
        print("Config file loaded successfully.")
    except FileNotFoundError:
        config.to_yaml(path)
        print("Config file not found. Saving default.")

    print("\nLoaded", config, "\n")
    return config


def main():

    config = load_cfg()
    torch.set_float32_matmul_precision("medium")

    
    model = MiniPilot(config)
    comset = CommaDataset(
        config.data.rootdir / config.data.sample,
        config.model.num_future_steps,
        transform=None,
    )

    samples = DataLoader(comset, batch_size=1, shuffle=False)
    
    world_tokens = model.prior_tokens(batch_size=1)
    for sample in tqdm(samples, desc="Processing samples"):
        intents = sample["intents"]
        sensors = sample["sensors"]
        labels = sample["labels"][0]

        _, images, clouds, telemetry = model.unpack_sensors(sensors)

        result = model(
            images=images,
            clouds=clouds,
            telemetry=telemetry,
            intents=intents,
            world_tokens=world_tokens,
        )

        preds = result["future_trajectory"][0]
        world_tokens = result["updated_world_tokens"]

        height = labels[:, 2].unsqueeze(-1)

        preds = torch.cat((preds, height), dim=-1)
       
        label = labels.detach().contiguous().cpu().numpy()
        pred = preds.detach().contiguous().cpu().numpy()

        image = (
            images[0]
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        
        draw_path(label[10:], image)
        # draw_path(pred[10:], image)

        cv2.imshow("image", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
