import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class CloudSEN12Dataset(Dataset):
    def __init__(self, data_dir: str, bands: list[int] = list(range(13))):
        self.bands = bands

        # Find all image files, derive mask paths from naming convention
        image_paths = sorted(Path(data_dir).glob("*_image.npy"))

        self.samples = [
            {
                "image": p,
                "mask": Path(str(p).replace("_image.npy", "_mask.npy"))
            }
            for p in image_paths
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        # Load - rasterio saves as (C, H, W) already, no permute needed
        image = np.load(sample["image"]).astype(np.float32)  # (13, 512, 512)
        mask = np.load(sample["mask"]).astype(np.int64)       # (512, 512)

        # Select bands - default is all 13, can pass subset e.g. [0,1,2,3]
        image = image[self.bands]

        # Normalise: Sentinel-2 raw values -> 0.0 to 1.0
        image = image / 10000.0

        # Convert to tensors - already (C, H, W), no permute needed
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
