# SegMentors
Benchmarking self-supervised and semi-supervised learning approaches for
2D pediatric brain tumor segmentation.

### Branch Overview
This branch provides a PyTorch pipeline to pretrain a 2D U-Net with masked
reconstruction and fine-tune it for medical image segmentation on preprocessed
MRI `.npy` slices.

### Key Features and Usage
1. Self-supervised training: Use `run_training.py` to run the full pipeline,
   including reconstruction pretraining, supervised fine-tuning, checkpoint
   saving, and test-set prediction export.

2. Inference and visualization: Use `run_inference.py` to load a trained
   checkpoint, run prediction on a single `.npy` slice, and generate an
   overlay visualization against the ground-truth mask.

3. Notebook workflow: Use `run_training_colab.ipynb` to run the training
   pipeline in a notebook environment such as Google Colab.

### Directory Structure
- `/data`: Stores the train, validation, and test `.npy` images, masks, and
  split manifests such as `ssl_split_manifest_15.csv`,
  `ssl_split_manifest_20.csv`, and `ssl_split_manifest_25.csv`.

- `/outputs`: Stores training logs, checkpoints, loss curves, and exported test
  predictions.

- `/src`: Contains the core training modules, including dataset loading,
  pretraining, fine-tuning, U-Net definition, and utility functions.

- `run_training.py`: Main script for the end-to-end self-supervised and
  supervised training pipeline.

- `run_inference.py`: Helper functions for single-sample inference and
  prediction visualization.

- `run_training_colab.ipynb`: Notebook pipeline for running training in Colab.
