from huggingface_hub import snapshot_download


def download_cloudsen12(destination: str):
    print(f"Downloading CloudSEN12 to {destination}...")

    snapshot_download(
        repo_id="isp-uv-es/CloudSEN12Plus",
        local_dir=destination,
        repo_type="dataset",
        ignore_patterns=["*LQ*", "*MQ*", "*.csv", "*.json"]
    )

    print("Download complete.")


if __name__ == "__main__":
    download_cloudsen12("data/raw/cloudsen12")