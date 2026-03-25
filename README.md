# SegMentors

SegMentors is a modular PyTorch repository for self-supervised and semi-supervised 2D pediatric brain tumor segmentation. The current pipeline uses a 2D U-Net backbone, masked reconstruction pretraining on mixed labeled and unlabeled MRI slices, and supervised fine-tuning on the labeled subset to predict segmentation masks for:

- `ET`
- `NET`
- `CC`
- `ED`

The project is organized around preprocessed `.npy` slices where each image has shape `(4, H, W)` and each mask has shape `(H, W)`.

## Overview

The training workflow has two stages:

1. Self-supervised pretraining:
   The model reconstructs masked MRI content using the full U-Net encoder-decoder. The default pretext task masks modality-specific patches and asks the network to recover them from spatial and cross-modal context.
2. Supervised fine-tuning:
   The reconstruction head is replaced with a segmentation head, and the model is fine-tuned on the labeled training cases only.

The repository also includes fixed split manifests for different labeled-data regimes:

- `ssl_split_manifest_15.csv`
- `ssl_split_manifest_20.csv`
- `ssl_split_manifest_25.csv`

## Repository Structure

- [`start_training.py`](start_training.py): main training entry point
- [`load_dataset.py`](load_dataset.py): dataset loading, manifest filtering, and dataloaders
- [`pretraining.py`](pretraining.py): masking strategies and SSL pretraining loops
- [`fine_tuning.py`](fine_tuning.py): segmentation loss, validation, and prediction export
- [`unet2d.py`](unet2d.py): 2D U-Net backbone with a replaceable output head
- [`utils.py`](utils.py): shared utilities, metrics, checkpointing, and plotting
- [`make_ssl_split.py`](make_ssl_split.py): helper for building a labeled/unlabeled manifest
- [`ssl_unet2d_colab.ipynb`](ssl_unet2d_colab.ipynb): notebook version for Colab or Jupyter

## Data Layout

Expected folder structure:

```text
SegMentors/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
```

Expected file naming:

```text
BraTS-PED-00001-000__000000__img.npy
BraTS-PED-00001-000__000000__mask.npy
```

Where:

- image arrays are `(4, H, W)`
- mask arrays are `(H, W)`
- the case ID is the prefix before the first `__`

## Installation

Create an environment and install the required packages.

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install numpy matplotlib tqdm
```

## Training

Run the default pipeline:

```bash
python start_training.py
```

By default, this uses:

- `data/` as the dataset root
- `ssl_split_manifest_20.csv` as the labeled/unlabeled split
- `outputs/ssl_unet2d/` as the output directory

Example with a different manifest and output folder:

```bash
python start_training.py \
  --manifest-csv ssl_split_manifest_25.csv \
  --output-dir outputs/run_ssl_25
```

Useful arguments:

- `--pretrain-epochs`
- `--finetune-epochs`
- `--batch-size`
- `--pretrain-lr`
- `--finetune-lr`
- `--mask-ratio`
- `--patch-size`
- `--pretext-task`

Available pretext tasks:

- `cross_modal`
- `random_channel_dropout`

## Outputs

Each training run writes to the selected output directory and typically includes:

- `pretrain_best.pt`
- `finetune_best.pt`
- `training.log`
- `loss_curves.png`
- `test_predictions/`

`test_predictions/` contains predicted `.npy` segmentation masks for the test split.

## Jupyter / Colab

If you prefer notebooks, open:

- [`ssl_unet2d_colab.ipynb`](ssl_unet2d_colab.ipynb)

The notebook provides cells for:

- dependency installation
- optional Google Drive mounting
- path configuration
- argument configuration
- training execution
- output inspection

