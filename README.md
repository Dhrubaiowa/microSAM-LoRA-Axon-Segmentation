# microSAM-LoRA Axon Segmentation

This repository contains a deep learning pipeline for semantic segmentation of healthy and damaged axons in optic nerve histology images using parameter-efficient adaptation of **microSAM**.

The project uses a pretrained SAM ViT-B image encoder and applies **Low-Rank Adaptation (LoRA)** to the final transformer blocks while keeping the remaining encoder parameters frozen. A task-specific convolutional decoder is trained for three-class segmentation:

- Background
- Healthy axon
- Damaged axon

This work is part of my research on adapting vision foundation models for biomedical image segmentation with limited annotated data.

## Method

The main components of the pipeline are:

- microSAM / SAM ViT-B image encoder
- LoRA-based parameter-efficient fine-tuning
- Task-specific multiclass segmentation decoder
- Three-class healthy/damaged axon segmentation
- Cross-entropy and multiclass Dice loss
- Contact-aware loss weighting for touching axons
- Geometric and intensity-based data augmentation
- Patch-based training and sliding-window inference
- Gaussian blending of overlapping predictions
- Connected-component post-processing

Only the LoRA parameters and the task-specific segmentation decoder are optimized during training. The remaining pretrained encoder parameters are kept frozen.

## Repository Contents

The main implementation is provided in:

`microSAM_LoRA_Axon_Segmentation.ipynb`

The notebook includes:

1. Data loading and preprocessing
2. Multiclass label construction
3. Data augmentation
4. microSAM encoder initialization
5. LoRA integration
6. Task-specific segmentation decoder
7. Loss functions and evaluation metrics
8. Training and checkpointing
9. Sliding-window inference
10. Post-processing

## Data

The research dataset is not included in this repository.

The pipeline expects RGB histology images with separate annotations for healthy and damaged axons. These annotations are combined into a three-class semantic segmentation mask:

| Class | Label |
|---|---:|
| Background | 0 |
| Healthy axon | 1 |
| Damaged axon | 2 |

When healthy and damaged annotations overlap, the damaged-axon label is given priority.

## Model Configuration

The implementation uses the following main configuration:

- Input size: `512 × 512`
- Patch size: `512 × 512`
- Encoder: SAM ViT-B
- LoRA rank: `8`
- LoRA applied to the final four transformer blocks
- Number of output classes: `3`
- Optimizer: AdamW
- Loss: Cross-entropy + multiclass Dice
- Sliding-window inference with overlapping patches

These parameters can be modified in the configuration section of the notebook.

## Usage

Clone the repository and install the required Python packages.

Update the dataset and pretrained model paths in the configuration section of the notebook:

```python
DATA_ROOT = Path("path/to/axon_dataset")
MODEL_DIR = Path("path/to/pretrained/model")
```

The pretrained model weights and research dataset are not distributed with this repository.

After configuring the paths, the notebook can be run sequentially for data preparation, model training, validation, and inference.

## Requirements

The main dependencies are:

- Python
- PyTorch
- NumPy
- OpenCV
- SciPy
- pandas
- Pillow
- Matplotlib
- tqdm

Package dependencies are also provided in `requirements.txt`.

## Acknowledgments

This implementation builds on the **Segment Anything Model (SAM)** and **microSAM** ecosystem. The pretrained foundation-model components belong to their respective authors and projects.

The LoRA adaptation, task-specific segmentation pipeline, training strategy, and axon-analysis implementation in this repository were developed as part of my research in medical image analysis.

## Author

**Durjoy Deb Dhruba**  
Ph.D. Candidate, Electrical and Computer Engineering  
University of Iowa
