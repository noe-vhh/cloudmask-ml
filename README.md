# CloudMask ML

> Semantic segmentation prototype for cloud pixel detection in Sentinel-2 satellite imagery -
> with a path to per-sensor deployment via ONNX export into existing Java-based EO pipelines.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20%2B%20ROCm%206.2-orange)
![ONNX](https://img.shields.io/badge/Export-ONNX%201.21-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## What This Is

Cloud masking is a critical preprocessing step in every Earth Observation (EO) pipeline.
Cloudy pixels corrupt downstream analysis - crop health, flood extent, urban change - and
must be identified and removed before any science can happen.

This prototype asks: **can a small trained ML model replace a hand-tuned algorithmic approach,
generalise across sensors, and still be lightweight enough for constrained deployment?**

The answer is validated against [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloudless),
the industry-standard algorithmic baseline for Sentinel-2 cloud detection.

---

## Architecture

A **U-Net** semantic segmentation model with a **ResNet34** encoder backbone, preceded
by a learned **band projector** (1×1 convolution) for sensor agnosticism.

```
[Band Projector]  ←  per-sensor, maps N input bands → fixed internal channels
       ↓
[U-Net Encoder]   ←  shared core, learns abstract features at decreasing resolution
       ↓
[Bottleneck]
       ↓
[U-Net Decoder]   ←  reconstructs full resolution via skip connections
       ↓
Output mask       ←  per-pixel cloud probability → binary label (0=clear, 1=cloud)
```

**Sensor agnosticism:** each sensor gets its own projector (trained cheaply on ~100-200
labelled samples). The U-Net core is shared and improves with every sensor added.
Deployment is a single fused `.onnx` file per sensor - the Java pipeline never changes.

```
Sentinel-2 (13 bands) ─┐
Landsat-8  (11 bands) ─┤→ [Band Projector] → 32ch → [U-Net] → cloud mask
ClientSat  ( 4 bands) ─┘
```

---

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | PyTorch 2.5 + ROCm 6.2 | Industry standard, AMD GPU support via ROCm |
| Segmentation | segmentation-models-pytorch | Pre-built U-Net variants, clean API |
| Augmentation | Albumentations | Synced image+mask transforms, fast |
| Export | ONNX 1.21 | Portable format - bridges Python training to Java pipeline |
| Runtime | onnxruntime | Runs ONNX in Java, C++, Python with near-native speed |
| GPU | AMD RX 6800 XT (16GB) | Available hardware, ROCm verified |
| Dataset | CloudSEN12Plus (HQ) | 342 human-expert labelled Sentinel-2 patches |

---

## Project Structure

```
cloudmask-ml/
├── data/
│   └── download_cloudsen12.py   # Targeted HQ extraction via HTTP range requests
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Generic PyTorch Dataset - sensor agnostic
│   ├── train.py                 # Training loop with augmentation + checkpointing
│   ├── evaluate.py              # IoU/F1 metrics against val/test set
│   ├── export.py                # ONNX export and validation
│   └── predict.py               # Single image inference
├── models/
│   ├── cloudmask_unet_core.pth  # Shared U-Net core
│   ├── projectors/              # Per-sensor band projectors
│   └── onnx/                    # Fused deployment models
├── config.yaml                  # All hyperparameters and paths
├── SETUP.md                     # Full environment setup (ROCm, PyTorch)
├── NOTES.md                     # Concise project reference
├── LEARNING.md                  # Concepts and ELI5 explanations
└── requirements.txt             # Python dependencies
```

---

## Quick Start

See [SETUP.md](SETUP.md) for full environment setup including ROCm and PyTorch.

```bash
git clone https://github.com/noe-vhh/cloudmask-ml.git
cd cloudmask-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Extract the dataset (342 HQ samples, ~2GB):

```bash
python data/download_cloudsen12.py
```

Train:

```bash
python src/train.py
```

All hyperparameters and paths are configured in `config.yaml`.

---

## Dataset

**CloudSEN12Plus** - `isp-uv-es/CloudSEN12Plus` on HuggingFace.

We extract only the **HQ fixed subset** (human expert labels, reviewed in v1.1):

| Split | Samples |
|-------|---------|
| train | 267 |
| validation | 23 |
| test | 52 |
| **total** | **342** |

The full dataset is ~101GB. Targeted extraction via HTTP range requests pulls only
the 342 samples we need (~2GB total).

---

## Status

- [x] GPU compute stack (ROCm 6.4 + PyTorch 2.5, AMD RX 6800 XT verified)
- [x] Dataset extraction pipeline (342 HQ samples via HTTP range requests + rasterio)
- [x] Full extraction run (342 samples across train/validation/test)
- [x] config.yaml (model, training, data configuration)
- [x] src/dataset.py (lazy loading, augmentation, binary mask collapse)
- [x] src/train.py (training loop, transfer learning, checkpointing)
- [x] src/evaluate.py (IoU/F1 metrics)
- [ ] src/export.py (ONNX export and validation)
- [ ] src/predict.py (single image inference)
- [ ] Benchmark results against s2cloudless
- [ ] Cross-sensor evaluation (Landsat-8)

---

## Roadmap

**Prototype (current)**
Train and benchmark on Sentinel-2 CloudSEN12 HQ data. Validate against s2cloudless.
Export to ONNX. Demonstrate Java pipeline integration.

**Cross-sensor validation**
Evaluate zero-shot on Landsat-8 (USGS Biome dataset). Fine-tune band projector.
Quantify onboarding cost: labels needed, training time, accuracy delta.

**Production architecture**
Per-sensor projectors trained from shared core. Active learning labelling pipeline.
Periodic core retraining as multi-sensor data accumulates.

**On-board inference (research)**
Quantised ONNX models for constrained satellite hardware. Federated learning vision.

---

## Goal

Demonstrate that a lightweight ML approach can match or exceed the accuracy of a
hand-tuned algorithmic approach for cloud masking, while providing a clear path to deployment
across multiple sensors via ONNX export - without touching the existing Java pipeline.