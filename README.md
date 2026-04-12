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
├── data/               # Dataset scripts and download utilities
├── models/             # Model definitions and checkpoints
├── notebooks/          # Exploration and evaluation notebooks
├── src/                # Training, inference, and export scripts
├── tests/              # Unit and integration tests
├── SETUP.md            # Full environment setup and reproducibility guide
└── requirements.txt    # Python dependencies
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
- [x] dataset.py updated (binary mask collapse, dtype cleanup)
- [ ] Model training
- [ ] Benchmark validation against s2cloudless
- [ ] ONNX export and runtime validation

---

## Goal

Demonstrate ML value internally as a step toward per-sensor lightweight models
and a federated learning architecture for on-board satellite processing.