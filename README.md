# SegMentors

**Benchmarking Semi-Supervised Learning Approaches to Increase Medical Image Segmentation Performance

This ECE324 project seeks to better understand the impacts of semi-supervised learning in the task of medical segmentation. 


## Project Overview

## Repository Overview
This GitHub Repository all code pertaining to:
- Preprocessing raw data files
- Organizing and splitting data
- Training the baseline U-Net
- Visualizing model inference on the test set

By the conclusion of this project, the GitHub will include the semi-supervised methodologies outlined in the interim report. 

## Environment Setup
To install the environment, run:

```bash
cd /path/to/SegMentors
conda env create -f environment.yaml
conda activate unet2d-env
```

## Data & Train Pipeline

To preprocess the dataset, first download the raw images and masks from: 

```bash
https://www.cancerimagingarchive.net/collection/brats-peds/$0
```

The data should be stored under the directory
```bash
/path/to/SegMentors/raw_dataset
```

From which you may run:
```bash
python data_preprocessing.py
```


Additionally, the following script will append the processed masks into the correct folder structure and data split:

```bash
python data_split.py
```

Training can be executed by running:
```bash
python train_unet.py
```

Visualizing inference on test cases is effected by:
```bash
python inference.py
```


