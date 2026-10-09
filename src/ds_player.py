import glob
import os

import cv2

from datasets.comma2k19.comma2k19 import CommaDataset
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = "/mnt/ssd/Datasets/comma2k19"


def main():

    videos = glob.glob(
        os.path.join(ROOT_DIR, "**", "*.hevc"),
        recursive=True,
    )

    for video in tqdm(videos):
        video_path = Path(video)
        print(f"Processing video: {video_path}")

        dataset = CommaDataset(
            video_path.parent,
            1,
            transform=None,
        )

        for idx in range(0, len(dataset), 50):
            sample = dataset[idx]
            image = sample["sensors"]["camera_imgs"]
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow(f"Stream", image)
            cv2.waitKey(1)
            


if __name__ == "__main__":
    main()
