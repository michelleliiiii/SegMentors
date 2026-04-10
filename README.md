# SegMentors Teacher-Student Branch

This branch contains the semi-supervised teacher-student training pipeline for 2D medical image segmentation on the BraTS pediatric dataset.

## What This Branch Runs

The workflow is:

1. Prepare the dataset into `data/train`, `data/val`, and `data/test`.
2. Create the semi-supervised split manifest in `ssl_split_manifest.csv`.
3. Train the teacher on the labeled training subset.
4. Train the student using labeled data plus teacher pseudo-labels on the unlabeled subset.
5. Optionally evaluate checkpoints and generate teacher probability-map visualizations.

## Environment Setup

From the repository root:

```bash
conda env create -f environment.yaml
conda activate unet2d-env
```

## Running the Teacher-Student Model

Run these commands from the repository root.

### 1. Train the teacher

```bash
python train_teacher.py
```

This writes the best teacher checkpoint to:

```bash
teacher_best.pt
```

### 2. Train the student

```bash
python train_student.py
```

By default, the student script:

- Loads `teacher_best.pt`
- Uses the labeled portion of `data/train`
- Uses teacher pseudo-labels for the unlabeled portion of `data/train`
- Saves the best student checkpoint to `student_best.pt`

Important dependency:

- `train_student.py` requires `teacher_best.pt` to already exist.

## Evaluation

Evaluate the teacher:

```bash
python evaluate.py --ckpt teacher_best.pt --split val --output-dir "your output directory here"
```

Evaluate the student:

```bash
python evaluate.py --ckpt student_best.pt --split val --output-dir "your output directory here"
```

## Optional Utilities

Generate teacher probability-map visualizations:

```bash
python run_teacher_probability_maps.py
```

Run teacher hyperparameter search:

```bash
python random_search_teacher.py
```

Run student hyperparameter search:

```bash
python random_search_student.py --trials 10
```



## Expected Files

After a standard run, the main artifacts are:

- `ssl_split_manifest.csv`
- `teacher_best.pt`
- `student_best.pt`

## AI Statement

OpenAI Codex was used in this branch for generating code and for preparing project documentation, including this README. All generated content has been reviewed and validated by the project authors before submission and deployment.
