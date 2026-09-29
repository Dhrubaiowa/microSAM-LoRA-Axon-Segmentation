# microSAM-LoRA-Axon-Segmentation

This repository contains a deep learning pipeline for semantic segmentation of healthy and damaged axons in optic nerve histology images using parameter-efficient adaptation of **microSAM**.

The project uses a pretrained SAM ViT-B image encoder and applies **Low-Rank Adaptation (LoRA)** to the final transformer blocks while keeping the remaining encoder parameters frozen. A task-specific convolutional decoder is trained to perform three-class segmentation:

- Background
- Healthy axon
- Damaged axon

This work is part of my research on adapting vision foundation models to biomedical image segmentation with limited annotated data.

## Method

The main components of the pipeline are:

- microSAM / SAM ViT-B image encoder
- LoRA-based parameter-efficient fine-tuning
- Task-specific multiclass segmentation decoder
- Three-class healthy/damaged axon segmentation
- Cross-entropy + multiclass Dice loss
- Contact-aware loss weighting for touching axons
- Offline geometric and intensity augmentation
- Overlapping patch-based training and inference
- Gaussian blending during sliding-window inference
- Connected-component post-processing for small-object removal

Only the LoRA parameters and segmentation decoder are optimized during training; the remaining pretrained encoder parameters are frozen.

## Repository Contents

`microSAM_LoRA_Axon_Segmentation.ipynb`

The notebook includes:

1. Data loading and preprocessing
2. Multiclass label construction
3. Data augmentation
4. microSAM encoder initialization
5. LoRA integration
6. Segmentation decoder
7. Loss functions and evaluation metrics
8. Training and checkpointing
9. Sliding-window inference
10. Post-processing

## Data

The dataset used in this research is not included in this repository.

The pipeline expects RGB histology images with separate annotations for healthy and damaged axons. The annotations are combined into a three-class semantic mask:

| Class | Label |
|---|---:|
| Background | 0 |
| Healthy axon | 1 |
| Damaged axon | 2 |

When healthy and damaged annotations overlap, the damaged-axon label is given priority.

## Model Configuration

The current implementation uses:

- Input/patch size: `512 × 512`
- Encoder: SAM ViT-B
- LoRA rank: `8`
- LoRA applied to the final four transformer blocks
- Number of output classes: `3`
- Optimizer: AdamW
- Loss: Cross-entropy + multiclass Dice
- Sliding-window inference with overlapping patches

These settings correspond to the configuration used in the included research implementation and can be modified in the configuration section of the notebook.

## Usage

Clone the repository and install the required dependencies.

Update the dataset and pretrained model paths in the configuration section of the notebook:

```python
DATA_ROOT = Path("path/to/axon_dataset")
MODEL_DIR = Path("path/to/pretrained/model")
