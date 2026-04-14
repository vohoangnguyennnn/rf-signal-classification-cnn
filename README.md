# RF Signal Classification via Lightweight CNN

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![NumPy](https://img.shields.io/badge/numpy-1.23-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)


A custom CNN-based RF signal classification pipeline operating on spectrogram images. The model is kept under **100k parameters** using MobileNetV2-style inverted residuals, Squeeze-and-Excitation attention, and multi-scale feature fusion — designed to train fast on limited hardware while maintaining competitive accuracy.

---

## Overview

This project classifies radio frequency (RF) signal spectrograms into 12 modulation/signal-type categories. It is engineered for:

- **Parameter efficiency** — sub-100k trainable weights, suitable for edge deployment.
- **Robust training** — Mixup augmentation, OneCycleLR, gradient clipping, and early stopping reduce overfitting without requiring massive datasets.
- **Reproducibility** — all RNG sources are seeded; `cudnn.deterministic` is enforced.
- **Portable inference** — a TorchScript checkpoint is exported for deployment outside the training environment.

---

## Model Architecture

### SignalBackbone (`< 100k params`)

| Stage | Block | Channels | Stride | SE Attention |
|-------|-------|----------|--------|--------------|
| Entry | Conv 3×3 + BN + ReLU6 | 16 | 2 | — |
| Stage 1 | Inverted Residual ×2 + IR + SE | 16 → 24 | 1 / 2 | No / Yes |
| Stage 2 | Inverted Residual ×2 + SE | 24 → 32 | 2 / 1 | Yes |
| Stage 3 | Inverted Residual ×3 + SE | 32 → 48 → 56 | 2 / 1 | Yes |
| Fusion | MultiScaleFusion (GAP + 1×1 projection) | — | — | — |
| Head | Dropout 0.4 → Linear | 56 → num_classes | — | — |

### Key Building Blocks

**`ChannelScaler`** — Squeeze-and-Excitation module. Global-average-pools the spatial dimension, passes through two FC layers (reduction ratio = 4), and re-weights channels with a Sigmoid gate.

**`ResidualMobileUnit`** — Inverted residual block with depthwise separable 3×3 convolution. When stride=1 and channels match, an identity skip connection is used; otherwise a strided path. SE attention is applied post-convolution.

**`MultiScaleFusion`** — Fuses mid-level (Stage 2) and deep (Stage 3) feature maps. Both are GAP-pooled to `[B, C, 1, 1]`, projected to a common channel depth, summed, normalized, and activated. This preserves information from earlier convolutional stages without requiring matching spatial resolutions.

---

## Training Pipeline

```
DataLoader (ImageFolder, 95/5 split)
    ↓
Train transform: ToTensor → RandomHFlip → RandomVFlip → ColorJitter
               → RandomGrayscale → RandomErasing → Normalize
Val transform:   ToTensor → Normalize
    ↓
Mixup (α = 0.3) applied per batch
    ↓
AdamW (lr=3e-3, weight_decay=0.02)
OneCycleLR (pct_start=0.12, cos annealing, div_factor=10, final_div_factor=500)
Gradient clipping (max_norm=1.0)
Early stopping (patience=30 epochs)
    ↓
Outputs: loss/accuracy plots · confusion matrix · per-class P/R/F1 log · TorchScript .pt
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Mixup augmentation** | Soft label interpolation between two randomly shuffled samples; λ drawn from Beta(α, α), mirrored to `max(λ, 1-λ)` for symmetry. |
| **OneCycleLR** | Warm-up to 12% of training, cosine annealing to `MAX_LR/500`. Smooths gradient descent and avoids late-sharp minima. |
| **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients, especially under Mixup. |
| **Early stopping** | Patience = 30 epochs; restores best-weight checkpoint on exit. |
| **SE attention** | Channel gating after every residual block in Stages 2 and 3. |
| **Multi-scale fusion** | Mid-level / deep feature merge before the classifier head. |
| **TorchScript export** | `torch.jit.trace` on a fixed-shape input; loads directly in inference scripts. |
| **Reproducibility** | `seed=42`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. |
| **Cross-environment** | Auto-detects Kaggle (`/kaggle/input`) vs local (`./data`); no config changes needed. |

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | `[placeholder]` |
| Parameter Count | `< 100,000` |
| Training Epochs (typical) | `~[placeholder] before early stopping` |
| Confusion Matrix | `outputs/plots/confusion_matrix.png` |

---

## Project Structure

```
rf-signal-classification-cnn/
├── src/
│   └── train.py        # Full training pipeline (self-contained)
├── notebooks/
│   └── code_kaggle.py  # Kaggle-specific training notebook (optional)
├── outputs/
│   └── Model.pt  # TorchScript checkpoint
└── requirements.txt
```

> **Note:** The dataset is not included in this repository. Place your spectrogram image folders under `./data` (see Dataset Instructions below).

---

## Installation

```bash
# Clone the repository
git clone git@github.com:vohoangnguyennnn/rf-signal-classification-cnn.git
cd rf-signal-classification-cnn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0
torchvision>=0.15
numpy
matplotlib
seaborn
scikit-learn
```

---

## How to Run

### Local

```bash
python src/train.py
```

The script automatically uses `./data` as the dataset root and `cuda:0` if a GPU is available (falls back to CPU).

### Kaggle

Place the dataset in the Kaggle input directory:

```
/kaggle/input/datasets/huynhthethien/radarcommunsignaldata2026train/
```

Then run:

```bash
python src/train.py
```

The `locate_dataset_root()` helper in `train.py` detects the Kaggle environment and adjusts the data path automatically.

---

## Dataset Instructions

The model expects **ImageFolder-style** data:

```
./data/
├── BPSK/
│   ├── img_001.png
│   └── img_002.png
├── QPSK/
│   └── ...
├── 8PSK/
│   └── ...
...
```

- Each subdirectory name is the class label.
- Images are loaded with `torchvision.datasets.ImageFolder`; formats supported by PIL (PNG, JPG, etc.) work out of the box.
- The script splits data **95% train / 5% validation** with a fixed seed (42).
- No dataset is included here — download and place it under `./data/` before running.

---

## Outputs

After a successful run, the following artifacts are written to `outputs/`:

| File | Description |
|------|-------------|
| `plots/loss_plot.png` | Training and validation loss curves per epoch. |
| `plots/accuracy_plot.png` | Training and validation accuracy curves; includes a 90% reference line. |
| `plots/confusion_matrix.png` | Heatmap of true vs. predicted labels. |
| `logs/best_epoch_log.txt` | Per-class precision, recall, F1-score, and support for every new best-accuracy epoch. |
| `{GroupID}_DeepLearning_Project_TrainedModel.pt` | TorchScript-traced model for inference. |

---

## Reproducibility

Every run is fully deterministic when the environment is identical:

```python
lock_every_seed(seed_value=42)   # random, np, torch, CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark    = False
```

The 95/5 train/val split also uses a seeded `random_split` generator.

---

## Future Improvements

- [ ] Swap ImageFolder split for a `Subset` with an explicit CSV-based annotation file for fine-grained control over class distribution.
- [ ] Add W&B / MLflow logging for experiment tracking.
- [ ] Experiment with cosine-annealing with warm restarts (`CosineAnnealingWarmRestarts`) as an alternative scheduler.
- [ ] Implement SWA (Stochastic Weight Averaging) in the final 25% of training for flatter minima.
- [ ] Add ONNX export alongside TorchScript for ONNX Runtime compatibility.
- [ ] Extend to multi-GPU training via `DistributedDataParallel`.
- [ ] Replace the fixed 95/5 split with stratified K-fold cross-validation.
