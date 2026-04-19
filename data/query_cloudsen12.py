import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/cloudsen12")

splits = ["train", "validation", "test"]
frames = []

for split in splits:
    df = pq.read_table(DATA_DIR / split / "metadata.parquet").to_pandas()
    # Add a column so we know which split each row came from after combining
    df["split"] = split
    frames.append(df)

# Combine all three splits into one DataFrame
combined = pd.concat(frames, ignore_index=True)

# Count samples grouped by split, label quality, and fixed flag
counts = combined.groupby(["split", "label_type", "fixed"]).size()

hq = combined[combined["label_type"] == "high"]
info = hq.groupby(["equi_zone", "fixed"]).size()

print(counts)
print(info)
print(hq.groupby("fixed")[["thick_percentage", "thin_percentage", "cloud_shadow_percentage", "clear_percentage"]].mean())
# print(combined.columns.tolist())
