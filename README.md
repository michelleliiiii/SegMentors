# SegMentors

**Benchmarking Semi-Supervised Learning Approaches to Increase Medical Image Segmentation Performance**

This ECE324 project seeks to better understand the impacts of semi-supervised learning in the task of medical segmentation. 

## Repository Overview

Pre-requisites:
- The dataset has been setup as per the main README.md
- A baseline U-Net has been trained on 20% labelled data as per the main README.md

## Train Pipeline

First, we require bounding boxes for the train and validation set:

```bash
cd /path/to/SegMentors
python bbox_inference.py --split train --seed 42
python bbox_inference.py --split val --seed 42
```

Next, we must infer the bounding boxes for the unlabelled train set. 

```bash
cd /path/to/SegMentors
python bbox_inference --split unlabeled_train --ckpt ckpt.pt --seed42
```

Open ```bash medsam.ipynb``` in colab. It can run locally, provided CUDA exists, but the paths will need to be adjusted. 

Follow the instructions in medsam.ipynb

Extract the pseudolabel masks to:
```bash
/path/to/SegMentors/data/train/pseudo_labels
```

From which you may run:
```bash
cd /path/to/SegMentors
python finetune_unet.py --ckpt ckpt.pt --seed 42
```

Inferencing on test set:
```bash
cd /path/to/SegMentors
python inference.py --ckpt ckpt.pt --seed 42 
```

Now you can benchmark against the baseline at the SOTA. 
