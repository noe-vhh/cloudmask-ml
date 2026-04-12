# CloudMask ML - Project Knowledge Base

## 1. What is this project?

CloudMask ML is a machine learning prototype that identifies cloud pixels in satellite imagery - specifically Sentinel-2 images. Every pixel in a satellite image gets labelled: cloud, or not cloud.

This matters because clouds corrupt downstream analysis. If you're measuring crop health, flood extent, or urban change, a cloudy pixel gives you garbage data. Removing it first is called **cloud masking**, and it's a critical preprocessing step in every Earth Observation (EO) pipeline.

The current approach at the company uses a hand-tuned algorithm written in C. This prototype asks: can a small trained ML model do it better, generalise across more sensors, and still be lightweight enough to run on a satellite?

---

## 2. Semantic Segmentation - what the model actually does

There are different levels of computer vision:

- **Image classification** - "is this image cloudy?" (one answer per image)
- **Object detection** - "where are the clouds?" (bounding boxes)
- **Semantic segmentation** - "which exact pixels are cloud?" (one label per pixel)

We need semantic segmentation. The output is a **mask** - an image the same size as the input, where each pixel holds a class label: 0 = clear, 1 = cloud.

---

## 3. The U-Net Architecture

U-Net is the dominant architecture for semantic segmentation, especially in scientific imaging. It gets its name from its U-shaped structure.

### How it works

**Encoder (left side of the U - zooming out):**
The model looks at the image at progressively lower resolutions. At each step it learns more abstract features - first edges, then textures, then "this region looks like cloud material." It's compressing the image into a small, rich summary.

**Bottleneck (bottom of the U):**
The most compressed representation. The model understands the scene globally here but has lost spatial precision.

**Decoder (right side of the U - zooming back in):**
The model reconstructs full resolution, informed by what it learned. At each zoom-in step it also receives corresponding zoom-out features via skip connections.

**Skip connections:**
Direct links from each encoder layer to its matching decoder layer. They pass fine spatial detail forward - "there was a hard edge here" - that would otherwise be lost in compression. This is what makes U-Net so effective for precise pixel-level tasks.

**Output:**
A mask the same size as the input. Each pixel has a score - cloud probability - which gets thresholded into a binary label.

```
Input (256×256×N_bands)
      ↓
  [Encoder] → features at 128px, 64px, 32px, 16px
      ↓
 [Bottleneck]
      ↓
  [Decoder] ← skip connections from encoder
      ↓
Output mask (256×256×1)
```

We use **segmentation-models-pytorch (smp)** which provides U-Net and other architectures pre-built. You configure the encoder backbone, input channels, and output classes - it handles the rest.

---

## 4. Sensor Agnosticism - the band projector

### The problem
Different satellites have different sensors. Sentinel-2 has 13 bands. A Dragonfly or Simera smallsat might have 4, 6, or 12. If you hardcode "4 bands in" to your model, it breaks the moment someone feeds it a different sensor's image.

### The solution - Option B: learned band projection
A **1×1 convolution** sits before the U-Net. It learns to map N sensor-specific input bands into a fixed number of internal channels (e.g. 32) that the U-Net always sees.

Think of it as a translator: no matter what language (sensor) comes in, it gets translated into a common internal language before the model processes it.

```
Sensor A: 4 bands  ─┐
Sensor B: 7 bands  ─┤→ [Band Projector: 1×1 conv] → 32 channels → [U-Net]
Sensor C: 12 bands ─┘
```

Each sensor gets its own projector (different input size, different learned weights), but the U-Net weights are **shared** across sensors. You're teaching the model a universal internal representation of "what a pixel means."

### Why 1×1 convolution?
A 1×1 conv operates on one pixel at a time across all its band values. It's a learned weighted sum across channels - combining your N sensor-specific bands into a richer fixed-size representation. No spatial information is used at this step, only spectral.

### Deployment strategy
- **Prototype/benchmark:** train with all 7 Sentinel-2 bands for best accuracy
- **Production/smallsat:** swap projector for 4-band version, fine-tune, export
- **New sensor:** train a new projector, keep U-Net weights frozen or fine-tune lightly

---

## 5. Data - CloudSEN12

**CloudSEN12Plus** is a large-scale labelled dataset for cloud detection in Sentinel-2 imagery. Hosted on HuggingFace: `isp-uv-es/CloudSEN12Plus`.

Key facts:
- 10,000+ image patches
- Three quality tiers: HQ (high quality), MQ, LQ
- Labels generated using s2cloudless - which is also our benchmark target
- Each patch is 512×512 pixels
- Each patch has a matching binary cloud mask

We download only the **HQ subset** for training and benchmarking. This is controlled via `ignore_patterns` in the download script, which filters out MQ and LQ files.

---

## 6. The PyTorch Dataset Class

PyTorch requires data to be served through a `Dataset` class. Think of it as a vending machine contract:

- `__init__` - stock the menu (build a list of sample paths, load nothing)
- `__len__` - how many items are available?
- `__getitem__(index)` - dispense item N

### Why lazy loading?
CloudSEN12 HQ is ~20-30GB. Loading everything into RAM at startup would crash. Instead, each sample is loaded from disk on demand, one at a time, only when the training loop asks for it.

### What __getitem__ does per sample:
1. Look up the file paths for image and mask at this index
2. Load both `.npy` files from disk with `np.load()`
3. Cast image to `float32`, mask to `int64`
4. Select only the bands we want: `image[self.bands]`
5. Normalise image: divide by 10000.0 (Sentinel-2 max reflectance)
6. Convert both to PyTorch tensors with `torch.from_numpy()`
7. No permute needed — rasterio saves as (C, H, W) already
8. Return the tensor pair: `(image_tensor, mask_tensor)`

### Why (C, H, W)?
PyTorch convention. Channels first - all internal operations expect this order. NumPy loads as (H, W, C) because it thinks spatially. PyTorch thinks in channels.

### Why float32 for image, int64 for mask?
- Image values after normalisation are decimals like `0.732` - needs floating point
- Mask values are class labels: `0` or `1` - whole numbers only
- PyTorch loss functions specifically expect integer class labels - float masks cause crashes

---

## 7. Normalisation

Raw satellite band values are integers on different scales per sensor. Neural networks are sensitive to input scale - large raw values cause unstable gradients during training.

Normalisation converts everything to a common range. For Sentinel-2:
```
normalised = raw_value / 10000.0  →  range: 0.0 to 1.0
```

For multi-sensor work, per-band mean/std normalisation (z-scoring) is more robust - it accounts for each band having a different typical brightness range.

---

## 8. ONNX Export - the bridge to C

**ONNX (Open Neural Network Exchange)** is an open format for representing ML models. It's the bridge between the Python ML world and the C runtime world.

### Why it matters for this project
The company's existing EO processing pipelines are written in C for performance. A trained PyTorch model can't run in C directly. ONNX export serialises the model into a portable format that **ONNX Runtime** can execute - in C, C++, Python, or anywhere else.

### The workflow
```
Train in PyTorch → Export to .onnx → Load with onnxruntime in C pipeline
```

### Why this is the pitch
It means ML value doesn't require rewriting the C pipeline. You bolt on an ONNX model as a preprocessing step. The C engineers don't need to touch Python. The model runs at near-native speed.

---

## 9. Benchmarking - how we validate accuracy

### What is s2cloudless?
s2cloudless is an existing algorithmic cloud detector for Sentinel-2, maintained by Sentinel Hub. It's widely used and well-validated. Our goal: match or exceed its accuracy with a trained model.

### Metrics for segmentation

**IoU (Intersection over Union)** - the primary metric for segmentation tasks.
```
IoU = (pixels correctly labelled cloud) / (pixels labelled cloud by either model or ground truth)
```
Think of it as: how much does our predicted cloud region overlap with the true cloud region? 1.0 = perfect. 0.0 = no overlap.

**F1 Score (Dice coefficient)** - balances precision and recall.
- Precision: of the pixels we called cloud, how many actually were?
- Recall: of the actual cloud pixels, how many did we catch?
- F1 is the harmonic mean of both - punishes models that cheat by predicting everything as cloud

**Accuracy** - least useful for cloud masking because cloud pixels are rare. A model that predicts "clear" for every pixel could be 95% accurate but useless.

### Train / Validation / Test split
- **Training set** - model sees this data and learns from it
- **Validation set** - model never trains on this, used to tune hyperparameters and catch overfitting
- **Test set** - model never sees this until final evaluation, gives honest benchmark numbers

### Overfitting - ELI5
Overfitting is when the model memorises the training data instead of learning general patterns. Like a student who memorises past exam answers instead of understanding the subject - they fail on new questions.

Signs: training accuracy high, validation accuracy low. Fix: more data, regularisation, simpler model.

### Benchmarking workflow
```
1. Train on training set
2. Evaluate on validation set during training (monitor for overfitting)
3. When satisfied, run once on held-out test set
4. Compare IoU/F1 against s2cloudless on same test images
5. If results are promising → export to ONNX → validate ONNX output matches PyTorch output
6. Present to CTO
```

---

## 10. Project File Structure

```
cloudmask-ml/
├── data/
│   ├── download.py       # Downloads CloudSEN12 from HuggingFace
│   └── dataset.py        # PyTorch Dataset class for CloudSEN12
├── models/               # Model definitions and saved checkpoints
├── notebooks/            # Exploration and evaluation notebooks
├── src/                  # Training, inference, and export scripts
├── tests/                # Unit and integration tests
├── SETUP.md              # Full environment setup guide (ROCm, PyTorch)
├── NOTES.md              # This file
├── README.md             # Project overview
└── requirements.txt      # Python dependencies
```

---

## 11. Stack Decisions - and why

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | PyTorch 2.5 | Industry standard for research, flexible, ROCm support |
| Segmentation | segmentation-models-pytorch | Pre-built U-Net variants, clean API, well maintained |
| Augmentation | Albumentations | Fast, rich augmentation library for image/mask pairs |
| Export | ONNX 1.21 | Portable model format, bridges Python → C |
| Runtime | onnxruntime | Runs ONNX models efficiently in any language |
| GPU | AMD RX 6800 XT + ROCm | Available hardware, PyTorch ROCm wheel verified |
| Dataset | CloudSEN12Plus (HQ) | Large-scale, Sentinel-2 specific, s2cloudless labels |

---

## 12. CloudSEN12Plus - Real Dataset Structure

Discovered by inspection, not assumption. Always do this before writing pipeline code.

**Format:** `.mlstac` binary files - not `.npy`. Each split has one binary blob
per label type. Individual samples are byte-range slices defined by `begin` and
`length` columns in the metadata.

**Metadata:** Each split contains `metadata.parquet` with 25 columns including
`label_type`, `fixed`, `url`, `begin`, `length`, `datapoint_id`.

**Filtering for quality:**
- `label_type == 'high'` - human expert labels only
- `fixed == 1` - reviewed and corrected in CloudSEN12+ v1.1

**HQ fixed sample counts:**
| Split | Samples |
|-------|---------|
| train | 267 |
| validation | 23 |
| test | 52 |
| **total** | **342** |

**Why not download everything?**
Full dataset is ~101GB. We only need 342 samples for the prototype.
Targeted extraction from the binary file using byte offsets is the right approach.

**Libraries needed:**
- `pyarrow` - read `.parquet` metadata files
- `requests`, `aiohttp` - HTTP dependencies for tacoreader streaming

## 13. .mlstac Format — Discovered by Inspection

Never assume a dataset format. Always inspect before writing pipeline code.

### File structure

Each .mlstac file contains three sections in order:
- A 10-byte binary header
- A JSON index mapping datapoint_id to [relative_offset, length]
- Sequential JPEG2000 encoded samples

### Finding the boundary

The JSON index ends where binary data begins. JPEG2000 files always start with
magic bytes 0x0000000c. We search for these in the first ~700KB of the file to
find the exact boundary position. This boundary is different per file so it must
be detected dynamically, not hardcoded.

### Offset resolution

Offsets in the JSON index are relative to the boundary, not absolute from the
start of the file. To get the absolute byte position:

    absolute_offset = boundary + relative_offset

### Sample format

Each sample is a JPEG2000 file containing 15 bands at 512x512 pixels, uint16.

- Bands 0-12: Sentinel-2 L1C spectral bands
- Band 13: Human cloud label (primary) — values 0=clear, 1=thick cloud, 2=thin cloud, 3=shadow
- Band 14: Secondary label

### Extraction strategy

1. Fetch first ~700KB of .mlstac file
2. Search for JPEG2000 magic bytes to find boundary position
3. Fetch bytes 0 to boundary and parse as JSON index
4. For each HQ fixed sample, look up datapoint_id to get relative offset and length
5. HTTP range request for absolute byte range
6. Write to temp .jp2 file, open with rasterio, read as numpy array
7. Split bands: image = data[:13], mask = data[13]
8. Save as {datapoint_id}_image.npy and {datapoint_id}_mask.npy

### Why not tacoreader

tacoreader streams the entire file footer remotely and times out on 22GB files.
We bypass it entirely using raw HTTP range requests and rasterio.

### Output structure

    data/extracted/
        train/
            ROI_xxxx__..._image.npy
            ROI_xxxx__..._mask.npy
        validation/
            ...
        test/
            ...