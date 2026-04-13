# CloudMask ML

Semantic segmentation prototype for cloud pixel detection in Sentinel-2 satellite imagery.

Built to validate whether a lightweight ML approach can match or exceed the accuracy of
the current algorithmic C implementation, with a path to ONNX export for integration
into existing C-based processing pipelines.

---

## Background

Cloud masking is a critical preprocessing step in Earth Observation (EO) pipelines -
cloudy pixels corrupt downstream analysis. The current approach uses a hand-tuned
algorithm implemented in C. This prototype explores whether a small trained segmentation
model can improve accuracy and generalise better across sensor conditions, while remaining
lightweight enough for constrained deployment environments.

Validated against the [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloudless)
benchmark on public Sentinel-2 data.

---

## Stack

| Component | Choice |
|-----------|--------|
| Framework | PyTorch 2.5 + ROCm 6.2 |
| Model | segmentation-models-pytorch |
| Augmentation | Albumentations |
| Export | ONNX |
| GPU | AMD Radeon RX 6800 XT |

---

## Project Structure

```
cloudmask-ml/
├── data/
│   └── download_cloudsen12.py   # CloudSEN12-specific extraction via HTTP range requests
├── src/
│   ├── __init__.py              # Makes src/ importable as a package
│   ├── dataset.py               # Generic PyTorch Dataset - agnostic to data source
│   ├── train.py                 # Training loop with augmentation and checkpointing
│   ├── evaluate.py              # Metrics against validation/test set
│   ├── export.py                # Export trained model to ONNX
│   └── predict.py               # Inference on a single image (demos)
├── models/                      # Saved model checkpoints
├── notebooks/                   # Exploration and evaluation notebooks
├── tests/                       # Unit and integration tests
├── config.yaml                  # All hyperparameters and paths - single source of truth
├── SETUP.md                     # Full environment setup guide (ROCm, PyTorch)
├── NOTES.md                     # Project knowledge base
├── README.md                    # Project overview
└── requirements.txt             # Python dependencies
```

---

## Quick Start

See [SETUP.md](SETUP.md) for full environment setup including ROCm and PyTorch installation.

```bash
git clone https://github.com/noe-vhh/cloudmask-ml.git
cd cloudmask-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Status

- [x] GPU compute stack (ROCm 6.4 + PyTorch 2.5, AMD RX 6800 XT verified)
- [x] Dataset extraction pipeline (342 HQ samples, HTTP range requests + rasterio)
- [x] Full extraction run (342 samples across train/validation/test)
- [x] config.yaml (model, training, and data configuration)
- [x] src/dataset.py (binary mask collapse, augmentation support, dtype cleanup)
- [x] src/train.py (training loop, augmentation, checkpointing)
- [ ] src/evaluate.py (IoU/F1 metrics against val/test set)
- [ ] src/export.py (ONNX export and runtime validation)
- [ ] src/predict.py (single image inference for demos)
- [ ] Benchmark validation against s2cloudless

---

## Goal

Demonstrate ML value internally as a step toward per-sensor lightweight models
and a federated learning architecture for on-board satellite processing.