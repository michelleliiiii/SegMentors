# SegMentors
Benchmarking Semi-Supervised Learning Approaches to Increase Medical Image Segmentation Performance

### Branch Overview
This branch provides a pipeline to train and evaluate nnUnet on preprocessed dataset.

### Key features and Usage
1. Data reformat: Use `data_convert_nnunet.py` to convert .npy data files into NIFTI format required by nnUnet

2. Run nnUnet Pipeline: Run `run.ipynb` to preprocess dataset, train and evaluate nnUnet on Google Colab.

### Directory Structure
- `/nnUNet_results`: Stores training result for nnUnet.

- `data_convert_nnunet.py`: Functions for converting data to nnUnet required format

- `dice_results.json`: Evaluation result on test set.

- `run.ipynb`: Pipeline for running nnUnet on Google Colab.