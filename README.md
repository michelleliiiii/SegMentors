# SegMentors

**Benchmarking Semi-Supervised Learning Approaches to Increase Medical Image Segmentation Performance

This ECE324 project seeks to better understand the impacts of semi-supervised learning in the task of medical segmentation. 


## Project Overview


## Repository Overview


The main page of this GitHub Repository provides an overview of:
- Preprocessing raw data files
- Organizing and splitting data
- Training the baseline U-Net
- Performing inference on the test set

**To reproduce the SOTA, and any methodologies used in the project, please switch to the appropriate branch and check the README posted respectively.**


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
https://drive.google.com/file/d/1ToWIOYik2N46K_0CHBOfxw-WXWn9KkN5/view?usp=share_link
```

The data should be stored under the directory
```bash
/path/to/SegMentors/raw_data
```

From which you may run:
```bash
cd eda
python data_preprocessing.py
```


Additionally, the following script will append the processed masks into the correct folder structure and data split:

```bash
python data_split.py
```


To simulate the data scarce regime, run the following script with the argument corresponding to the desired percentage of retained ground truths. Ex: Retain 20% ground truths:

```bash
python make_ssl_split.py --labeled_frac 0.2
```

To train a fully supervised baseline, simply retain all ground truths: 

```bash
python make_ssl_split.py --labeled_frac 1.0
```

Training can be executed by running:
```bash
python train_unet.py --seed 42
```

Inferencing on the test set is achieved by running:
```bash
python inference.py --ckpt ckpt.pt --seed 42
```



