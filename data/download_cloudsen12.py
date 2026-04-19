import requests
import rasterio
import numpy as np
import pyarrow.parquet as pq
import json
import tempfile
import os
from pathlib import Path
from tqdm import tqdm

# Parquet metadata path
DATA_DIR = Path("data/raw/cloudsen12")
# Extracted .npy files path
OUT_DIR = Path("data/extracted/cloudsen12")
# How far into the file to search for the JSON boundary
INDEX_FETCH_SIZE = 700_000
# JPEG2000 magic bytes - marks start of binary data, end of JSON index
JP2_MAGIC = b'\x00\x00\x00\x0c'

def find_json_boundary(url: str) -> int:
    """
    Fetch the start of a .mlstac file and find where the JSON index ends
    """
    r = requests.get(url, headers={"Range": f"bytes=0-{INDEX_FETCH_SIZE}"}, allow_redirects=True)
    pos = r.content.index(JP2_MAGIC)
    return pos

def fetch_index(url: str, boundary: int) -> dict:
    """
    Fetch and parse the JSON sample index from a .mlstac file
    """
    r = requests.get(url, headers={"Range": f"bytes=0-{boundary-1}"}, allow_redirects=True)
    content = r.content
    # Skip magic bytes at start, find first {
    start = content.index(b'{')
    return json.loads(content[start:].decode("utf-8"))

def fetch_sample(url: str, offset: int, length: int) -> np.ndarray:
    """
    Fetch one sample by absolute byte range and return as numpy array (15, 512, 512)
    """
    r = requests.get(
        url,
        headers={"Range": f"bytes={offset}-{offset+length-1}"},
        allow_redirects=True
    )
    # Write to a temp file because rasterio needs a seekable file
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as f:
        f.write(r.content)
        tmp_path = f.name
    try:
        with rasterio.open(tmp_path) as ds:
            # shape: (15, 512, 512)
            data = ds.read()
    finally:
        # always clean up temp file
        os.unlink(tmp_path)
    return data

def extract_split(split: str):
    """
    Extract all HQ samples for one split and save as .npy pairs
    """
    print(f"\n--- {split} ---")

    # Load and filter metadata
    parquet_path = DATA_DIR / split / "metadata.parquet"
    df = pq.read_table(parquet_path).to_pandas()
    hq = df[(df["label_type"] == "high")]
    print(f"HQ samples (all): {len(hq)}")

    # Output folder for this split
    out_split = OUT_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Each split may have multiple .mlstac files, process per unique URL
    for url, group in tqdm(hq.groupby("url"), desc=f"{split} files", unit="file"):
        print(f"Processing {url.split('/')[-1]} ({len(group)} samples)")

        # Find JSON boundary and fetch index - once per .mlstac file
        boundary = find_json_boundary(url)
        index = fetch_index(url, boundary)
        print(f"  Index boundary: {boundary}, entries: {len(index)}")

        for _, row in tqdm(group.iterrows(), desc="  samples", unit="sample", total=len(group), leave=False):
            did = row["datapoint_id"]
            out_image = out_split / f"{did}_image.npy"
            out_mask = out_split / f"{did}_mask.npy"

            # Skip if already extracted (allows resuming interrupted runs)
            if out_image.exists() and out_mask.exists():
                print(f"  Skipping {did} (already exists)")
                continue

            if did not in index:
                print(f"  WARNING: {did} not in index, skipping")
                continue

            rel_offset, length = index[did]
            abs_offset = boundary + rel_offset

            print(f"  Extracting {did}...")
            data = fetch_sample(url, abs_offset, length)

            # Split bands: 0-12 = spectral image, 13 = human cloud label
            # shape: (13, 512, 512), uint16
            image = data[:13]
            # shape: (512, 512), uint16

            mask = data[13]

            np.save(out_image, image)
            np.save(out_mask, mask)

    print(f"  Done.")

if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        extract_split(split)
