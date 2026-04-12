import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class CloudSEN12Dataset(Dataset):
    def __init__(self, data_dir: str, bands: list[int]):
        self.bands = bands
        
        # Path().glob() finds all files matching the pattern, sorted() ensures consistent ordering across runs
        image_paths = sorted(Path(data_dir).glob("*_image.npy"))
        
        # Build list of dicts
        self.samples = [
            {
                "image": p,
                # Only relevant if project has consistent naming conventions
                "mask": Path(str(p).replace("_image.npy", "_mask.npy"))
            }
            for p in image_paths
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        
        # Load
        # Shape: (H, W, C)
        image = np.load(sample["image"]).astype(np.float32)
        # Shape: (H, W)
        mask = np.load(sample["mask"]).astype(np.int64)
        
        # Select only the bands we want, e.g. [0,1,2,3] for 4-band mode
        image = image[:, :, self.bands]
        
        # Normalise: scale raw int values to 0.0–1.0 (Sentinel-2 max reflectance value is 10000)
        image = image / 10000.0
        
        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        # Fix axis order: (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)
        
        # Return Tensor Pair
        return image, mask