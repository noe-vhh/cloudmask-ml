import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class CloudSEN12Dataset(Dataset):
    def __init__(self, data_dir: str, bands: list[int] = list(range(13)), transform=None, sensor_max_reflectance: float = 10000.0):
        self.bands = bands
        self.transform = transform
        self.sensor_max_reflectance = sensor_max_reflectance

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

        # Load - rasterio saves as (C, H, W)
        # (13, 512, 512)
        image = np.load(sample["image"]).astype(np.float32)
        # (512, 512)
        mask = np.load(sample["mask"])

        # Select bands - default is all 13, can pass subset e.g. [0,1,2,3]
        image = image[self.bands]

        # Normalise: divide by sensor max reflectance (Sentinel-2 = 10000.0)
        image = image / self.sensor_max_reflectance

        # collapse to binary: 0=clear, 1=cloud
        mask = (mask > 0).astype(np.int64)

        #Albumentation
        if self.transform:
            #Transpose to the expected Albumentation format
            image = image.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)

            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            #Transpose to original format
            image = image.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)

        # Convert to tensors - already (C, H, W), no permute needed
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
