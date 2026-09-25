# MetENet-CNN-Image-Classification-MATLAB-Python
MetENet is a custom Convolutional Neural Network architecture designed for image classification, implemented in both MATLAB and Python (PyTorch). This repository covers the full experimental pipeline: architecture design, training, hyperparameter tuning, cross-validation, transfer learning comparisons, and multi-dataset evaluation.
# MetENet: Custom CNN Image Classification

**MetENet** is a custom Convolutional Neural Network architecture designed for image classification, implemented in both **MATLAB** and **Python (PyTorch)**. This repository covers the full experimental pipeline: architecture design, training, hyperparameter tuning, cross-validation, transfer learning comparisons, and multi-dataset evaluation.

---

## Author

**Eya Sahli**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [MetENet Architecture](#metenet-architecture)
3. [Datasets](#datasets)
4. [Repository Structure](#repository-structure)
5. [Environment Setup](#environment-setup)
6. [MATLAB Implementation](#matlab-implementation)
7. [Python / PyTorch Implementation](#python--pytorch-implementation)
8. [Training Strategy](#training-strategy)
9. [Results](#results)
10. [Technologies](#technologies)
11. [License](#license)

---

## Project Overview

This project focuses on:

- Designing and evaluating **MetENet**, a custom VGG-style CNN architecture
- Training and evaluating MetENet on **CIFAR-10**, **Fashion-MNIST**, and **SVHN**
- Comparing MetENet against well-known pretrained CNN architectures (AlexNet, GoogLeNet, SqueezeNet, MobileNetV2, ResNet-18)
- Testing different data-splitting strategies and applying **5-fold cross-validation**
- Providing **Python (PyTorch) equivalents** of the MATLAB experiments as Jupyter Notebooks
- Providing both `.mlx` (Live Script) and `.m` (plain script) versions for usability

---

## MetENet Architecture

MetENet is a custom CNN inspired by VGG-style convolutional blocks. It uses progressively increasing filter counts and regularisation at each stage.

![MetENet Architecture](figures/architecture_metenet.png)

```
Input: 32 × 32 × 3

Block 1  — Conv(3×3, 32) → BN → ReLU → Conv(3×3, 32) → BN → ReLU → MaxPool(2×2) → Dropout(0.10)
Block 2  — Conv(3×3, 64) → BN → ReLU → Conv(3×3, 64) → BN → ReLU → MaxPool(2×2) → Dropout(0.20)
Block 3  — Conv(3×3,128) → BN → ReLU → Conv(3×3,128) → BN → ReLU → MaxPool(2×2) → Dropout(0.30)
Block 4  — Conv(3×3,256) → BN → ReLU → Conv(3×3,256) → BN → ReLU → MaxPool(2×2) → Dropout(0.30)

Fully Connected:
  Dense(1024) → ReLU → Dropout(0.35)
  Dense(128)  → ReLU → Dropout(0.35)
  Dense(10)   → Softmax
```

**Design choices:**
- Batch Normalisation after every convolution for training stability
- Increasing dropout per block to prevent over-fitting in deeper representations
- Two convolutional layers per block for richer feature extraction before downsampling

---

## Datasets

### CIFAR-10 (primary)

| Property | Value |
|---|---|
| Image size | 32 × 32 × 3 (colour) |
| Classes | 10 |
| Training images | 50,000 |
| Test images | 10,000 |

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

### Fashion-MNIST (additional)

Grayscale images of fashion items (clothing, shoes, bags). Used as a more challenging alternative to MNIST to evaluate MetENet's generalisation on a different visual domain.

### SVHN — Street View House Numbers (additional)

Real-world digit images extracted from street number photographs. Introduces additional challenges including natural-scene noise and class imbalance.

---

## Repository Structure

```
MetENet-CNN-Image-Classification-MATLAB/
│
├── src/
│   ├── 01_metenet_3_blocks.mlx          # MetENet with 3 conv blocks (MATLAB)
│   ├── 01_metenet_3_blocks.m
│   ├── 02_metenet_4_blocks.mlx          # MetENet with 4 conv blocks (MATLAB)
│   ├── 02_metenet_4_blocks.m
│   ├── 03_best_result_metenet.mlx       # Best configuration + tuning (MATLAB)
│   ├── 03_best_result_metenet.m
│   ├── 04_cross_validation.mlx          # 5-fold cross-validation (MATLAB)
│   ├── 04_cross_validation.m
│   ├── pretrained_models/
│   │   ├── fine_tuning/                 # Pretrained CNNs fine-tuned on CIFAR-10
│   │   └── full_training/              # Pretrained CNNs trained from scratch
│   └── additional_datasets/            # Fashion-MNIST and SVHN experiments
│
├── Python-PyTorch/
│   ├── MetENet_cifar10.ipynb            # MetENet on CIFAR-10 (PyTorch)
│   ├── ResNet18_cifar10.ipynb           # ResNet-18 transfer learning on CIFAR-10 (PyTorch)
│   └── MetENet_SVHN.ipynb               # MetENet on SVHN (PyTorch)
│
├── docs/
│   ├── initial_results.md               # Initial model comparison results
│   ├── cross_validation_and_full_training.md
│   └── additional_datasets.md          # Fashion-MNIST and SVHN results
│
├── MetENet.pth                          # Saved PyTorch model weights
├── .venv/                               # Python virtual environment (Python 3.14 + CUDA)
└── README.md
```

---

## Environment Setup

### MATLAB

- MATLAB R2023a or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Datasets are downloaded automatically by the Live Scripts

### Python / PyTorch

**Requirements:**
- Python 3.14
- PyTorch 2.12.0 + CUDA 12.6 (`torch`, `torchvision`)
- `scikit-learn`, `matplotlib`, `numpy`, `notebook`, `ipykernel`

**Setup using the included virtual environment:**

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Register the Jupyter kernel (first time only)
python -m ipykernel install --user --name metenet-env --display-name "MetENet (Python 3.14 + CUDA)"

# Launch Jupyter Notebook
jupyter notebook
```

Open the desired notebook from the `Python-PyTorch/` folder and select the **MetENet (Python 3.14 + CUDA)** kernel.

---

## MATLAB Implementation

The MATLAB experiments are organised as sequential Live Scripts (`.mlx`) with equivalent plain scripts (`.m`) for readability.

| Script | Description |
|---|---|
| `01_metenet_3_blocks` | MetENet with 3 convolutional blocks |
| `02_metenet_4_blocks` | MetENet with 4 convolutional blocks (improved) |
| `03_best_result_metenet` | Best configuration with optimised hyperparameters |
| `04_cross_validation` | 5-fold cross-validation on the best configuration |
| `pretrained_models/fine_tuning/` | AlexNet, GoogLeNet, SqueezeNet, MobileNetV2, ResNet-18 fine-tuned |
| `pretrained_models/full_training/` | Same architectures trained from random initialisation |
| `additional_datasets/` | MetENet applied to Fashion-MNIST and SVHN |

**Training environment (MATLAB):**

| Component | Specification |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i5-13450HX |
| RAM | 32 GB |
| GPU | NVIDIA RTX 3050 6 GB Laptop |

---

## Python / PyTorch Implementation

Three Jupyter Notebooks replicate the MATLAB experiments in PyTorch with identical architecture and comparable hyperparameters. All notebooks are in `Python-PyTorch/`.

### `MetENet_cifar10.ipynb`

Full MetENet training pipeline on CIFAR-10.

- **Sections:** Setup & Imports → Dataset Loading → Preprocessing → Architecture → Training → Evaluation → Confusion Matrix → Per-class Accuracy → Comparison Summary
- **Normalisation:** Zero-centre (mean/std computed on training split only, mirroring MATLAB's `zerocenter` InputLayer)
- **Optimiser:** Adam, lr = 0.001
- **Scheduler:** StepLR, factor = 0.7 every 4 epochs
- **Batch size:** 128 | **Max epochs:** 30 | **Early stopping patience:** 3

### `ResNet18_cifar10.ipynb`

ResNet-18 transfer learning from ImageNet weights on CIFAR-10, mirroring the MATLAB fine-tuning experiment.

- **Pre-trained weights:** `ResNet18_Weights.IMAGENET1K_V1`
- **Head replacement:** `fc → Linear(512, 10)`
- **Partial freeze:** first 50 % of parameterised modules frozen
- **Learning rates:** base layers 0.001, new FC 0.01 (10× factor, matching MATLAB's `WeightLearnRateFactor=10`)
- **Scheduler:** StepLR, factor = 0.5 every 3 epochs
- **Input:** 224 × 224 (bicubic resize), ImageNet normalisation
- **Early stopping patience:** 2

### `MetENet_SVHN.ipynb`

MetENet 4-block architecture applied to the SVHN dataset.

- Same architecture and hyperparameters as `MetENet_cifar10.ipynb`
- Uses `torchvision.datasets.SVHN` with automatic label 10→0 conversion
- Includes full evaluation: accuracy, confusion matrix, per-class breakdown

---

## Training Strategy

Both MATLAB and PyTorch implementations share the same core strategy:

| Component | Value |
|---|---|
| Optimiser | Adam |
| Batch size | 128 |
| Max epochs | 30 |
| Learning rate | 0.001 (initial) |
| LR schedule | Step decay |
| Regularisation | Batch Normalisation + Dropout |
| Early stopping | Yes (patience 2–3 epochs) |

---

## Results

### CIFAR-10 — Initial Architecture Comparison (Fine-Tuning / Transfer Learning)

| Model | Accuracy |
|---|---:|
| AlexNet | 80.07% |
| MetENet 3 Blocks | 82.00% |
| SqueezeNet | 85.79% |
| MetENet 4 Blocks | 86.04% |
| GoogLeNet | 87.79% |
| MobileNetV2 | 90.91% |
| ResNet-18 | 91.80% |

### CIFAR-10 — Full Training from Scratch (Pretrained Architectures)

| Model | Accuracy |
|---|---:|
| AlexNet | 76.79% |
| SqueezeNet | 83.36% |
| GoogLeNet | 84.20% |
| ResNet-18 | 90.68% |

> AlexNet shows signs of overfitting when trained from scratch on CIFAR-10.

### CIFAR-10 — Optimised MetENet

| Configuration | Accuracy |
|---|---:|
| MetENet 4 Blocks (best hyperparameters) | 86.55% |
| MetENet 4 Blocks — 5-Fold Cross-Validation (mean) | 86.62% |

The cross-validation result confirms that the 86.55% accuracy is stable and not due to a favourable random split.

### Additional Datasets

| Dataset | Model | Accuracy |
|---|---|---:|
| Fashion-MNIST | MetENet 4 Blocks | 92.72% |
| SVHN | MetENet 4 Blocks | 95.39% |

MetENet generalises effectively beyond CIFAR-10, achieving strong results on both grayscale (Fashion-MNIST) and real-world noisy (SVHN) datasets.

---

## MATLAB vs PyTorch — Implementation Notes

The MetENet architecture is identical across both frameworks. Small result differences may occur due to:

| Factor | Detail |
|---|---|
| Weight initialisation | Different default strategies per framework |
| Optimiser internals | MATLAB Adam vs PyTorch Adam numerical differences |
| Data normalisation | MATLAB `zerocenter` vs PyTorch computed mean/std |
| Random seed handling | Framework-level differences in shuffle order |
| GPU kernel precision | Minor floating-point differences at the CUDA level |

These variations are expected in cross-framework deep learning comparisons and do not indicate an implementation error.

---

## Technologies

| Category | Tools |
|---|---|
| Deep learning (MATLAB) | MATLAB Deep Learning Toolbox, Image Processing Toolbox |
| Deep learning (Python) | PyTorch 2.12.0+cu126, torchvision 0.27.0+cu126 |
| Notebooks | Jupyter Notebook (`.ipynb`), MATLAB Live Scripts (`.mlx`) |
| Evaluation | scikit-learn (confusion matrix, classification report) |
| Visualisation | matplotlib |
| Environment | Python 3.14, CUDA 12.6, NVIDIA RTX 3050 6 GB |

---

## License

This project is licensed under the MIT License.
