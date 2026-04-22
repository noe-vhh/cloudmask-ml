# CloudMask ML

> A sensor-agnostic cloud masking system - shared U-Net core trained on Sentinel-2, with lightweight per-sensor band projectors for new sensor onboarding from ~100-200 labelled samples. Designed for the small satellite industry where every client flies a different sensor.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20%2B%20ROCm%206.2-orange)
![ONNX](https://img.shields.io/badge/Export-ONNX%201.21-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## What This Is

Cloud masking is a critical preprocessing step in every Earth Observation (EO) pipeline. Cloudy pixels corrupt downstream analysis - crop health, flood extent, urban change - and must be identified and removed before any science can happen.

The core problem in the small satellite industry is not cloud masking itself - it's that **every client flies a different sensor**. Existing cloud masking solutions are hardcoded to specific sensors (Sentinel-2, Landsat-8). A small sat company flying a novel 4-band imager has no off-the-shelf solution, no large labelling budget, and no dedicated science team to build one.

This system is designed to solve that. **Sentinel-2 is the training vehicle for the shared core - not the product.** The value is delivered at the band projector layer, where any new sensor can be onboarded with ~100-200 labelled samples without retraining the core.

The core is validated against [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloudless) as a proof point. The headline result is the cross-sensor onboarding story: new sensor, ~100-200 labelled samples, production-quality masks.

---

## Architecture

A **U-Net** semantic segmentation model with a **ResNet34** encoder backbone, preceded
by a learned **band projector** (1×1 convolution) per sensor.

```
[Band Projector]  ←  per-sensor, maps N input bands → 32 channels
       ↓
[U-Net Encoder]   ←  shared core, learns abstract cloud structure
       ↓               frozen after Phase 1 training
[Bottleneck]
       ↓
[U-Net Decoder]   ←  reconstructs full resolution via skip connections
       ↓
Output mask       ←  per-pixel cloud probability → binary label (0=clear, 1=cloud)
```

**What the core learns:** spatial and structural characteristics of clouds - shapes,
textures, edges, morphology. Cloud structure is determined by atmospheric physics,
not by which sensor captured it. This knowledge transfers across sensors.

**What the projector handles:** spectral differences between sensors - different band
counts, different wavelengths, different radiometric calibration. Each sensor gets its
own projector trained cheaply on ~100-200 labelled samples.

```
Sentinel-2 (13 bands) ─┐
Landsat-8  (11 bands) ─┤→ [Band Projector] → 32ch → [Shared U-Net Core] → mask
ClientSat  ( 4 bands) ─┘
```

**Core integrity:** once trained, the core is frozen. Sensor-specific data only ever
touches the projector. The core improves periodically via joint retraining across all
accumulated sensor data - not per-sensor fine-tuning.

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
| Dataset | CloudSEN12Plus (HQ) | Sentinel-2 human-expert labelled patches - training vehicle for core |
| Experiment Tracking | Weights & Biases | Live loss curves, GPU metrics, hyperparameter logging |

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
│   ├── export.py                # ONNX export and validation (pending)
│   └── predict.py               # Single image inference (pending)
├── models/
│   ├── core/                    # Shared U-Net core checkpoints
│   ├── projectors/              # Per-sensor band projectors (pending)
│   └── onnx/                    # Fused deployment models, one per sensor
├── config.yaml                  # All hyperparameters and paths
├── SETUP.md                     # Full environment setup (ROCm, PyTorch)
├── NOTES.md                     # Live working reference - status, runs, next steps
├── LEARNING.md                  # How everything works - analogies, diagrams, mechanics
├── DECISIONS.md                 # Why decisions were made - alternatives considered
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

Extract the dataset (HQ samples):

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

We use the **HQ subset** (human expert labels, reviewed in v1.1). Sentinel-2 is the
training vehicle for the shared core - chosen because it is the best available public
dataset with expert human labels, not because the system is Sentinel-2-specific.

| Split | Samples (current) |
|-------|-------------------|
| train | 9,177 |
| validation | 612 |
| test | 1,060 |
| **total** | **10,849** |

The full dataset is ~101GB. Targeted extraction via HTTP range requests pulls only
the samples needed.

---

## Evaluation

Four-tier evaluation strategy:

| Tier | Setup | Purpose | Status |
|------|-------|---------|--------|
| 1 | In-distribution Sentinel-2 test set | Proof point vs s2cloudless | ✓ F1: 0.9198, IoU: 0.8515 (Run 2) |
| 2 | Zero-shot Landsat-8 (11 bands) | Raw cross-sensor transfer - diagnostic checkpoint | Pending |
| 3 | Fine-tuned Landsat-8 projector (11 bands) | Tier 2 vs Tier 3 delta = headline result | Pending |
| 4 | Fine-tuned projector on 4-band data (38-Cloud/95-Cloud) | Band-poverty test - small sat client reality | Pending |

The **headline result is the Tier 2 vs Tier 3 delta** - the empirical proof of the
sensor-agnosticism claim. Tier 1 (Sentinel-2) is a proof point, not the product story.

---

## Status

- [x] GPU compute stack (ROCm 6.2 + kernel 6.8.0-55 + PyTorch 2.5, AMD RX 6800 XT verified)
- [x] Dataset extraction pipeline (HQ samples via HTTP range requests + rasterio)
- [x] config.yaml (model, training, data configuration)
- [x] src/dataset.py (lazy loading, augmentation, binary mask collapse)
- [x] src/train.py (training loop, augmentation, transfer learning, checkpointing, W&B)
- [x] src/evaluate.py (IoU/F1/Precision/Recall/Accuracy)
- [x] First training run (20 epochs, ResNet34, batch_size 8)
- [x] Tier 1 baseline - F1: 0.7076, IoU: 0.5475 ✓ beats s2cloudless
- [x] dataset.py - A.Resize(512,512) for mixed resolution HQ samples
- [x] Full HQ data extraction verified (9,177 train / 612 val / 1,060 test)
- [x] Run 2 - full HQ dataset, 20 epochs, batch_size 16 ✓ F1: 0.9198, IoU: 0.8515
- [x] Training optimisations - mixed precision, persistent workers, cudnn.benchmark
- [ ] CosineAnnealingWarmRestarts + robustness augmentation + 100 epochs (Run 3 in progress)
- [ ] src/export.py (ONNX export and numerical validation)
- [ ] src/predict.py (single image inference, visualisation, CPU timing benchmark)
- [ ] CPU cost benchmark - ResNet34 vs MobileNetV2 vs INT8 vs 256×256 on laptop CPU
- [ ] Band Projector module (explicit per-sensor nn.Module)
- [ ] Tier 2 - zero-shot Landsat-8
- [ ] Tier 3 - fine-tuned Landsat-8 projector

---

## Roadmap

**Phase 1 - Core foundation (current)**
Full HQ data verified (10,849 samples). Implement CosineAnnealingWarmRestarts and
robustness augmentation (blur, noise, elastic distortion). Train core on best available
HQ data for 100 epochs. Core is then frozen - it is the foundation for everything else.

**Phase 2 - Deployment bridge**
ONNX export and numerical validation. Single image inference via `predict.py`.
CPU cost benchmarking: ResNet34 float32 → MobileNetV2 float32 → MobileNetV2 INT8 → MobileNetV2 INT8 256×256.
Each config measured for F1/IoU and CPU inference time on a Fargate-proxy CPU machine.
Java pipeline integration demonstration. AWS Fargate is CPU-only - the cloud deployment
target is MobileNetV2 INT8, not ResNet34. Measure before claiming.

**Phase 3 - Band projector**
Explicit per-sensor projector module. Sentinel-2 projector trained against frozen core.
Architecture properly separated.

**Phase 4 - Cross-sensor validation**
Tier 2: zero-shot Landsat-8 - first empirical test of core generalisation.
Tier 3: Landsat-8 projector fine-tuning - the headline result.
Tier 2 vs Tier 3 delta = the onboarding cost story.

**Phase 5 - Characterisation**
Generalisation bounds: what transfers, what doesn't, why. Honest small sat assessment.
Report / paper content.

**Phase 6 - Future**
Partial core unfreezing for severely degraded sensors. Core v2 joint retraining.
MQ data integration if diversity gap confirmed by Tier 2.

---

## Goal

Demonstrate a sensor-agnostic cloud masking system where a new sensor can be onboarded
with ~100-200 labelled samples, achieving production-quality masks without retraining
the shared core - with a clear path to deployment via ONNX export into existing
Java-based EO pipelines.
