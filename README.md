# SegMentors
Benchmarking Semi-Supervised Learning Approaches to Increase Medical Image Segmentation Performance

### Branch Overview
This branch provides a streamlined pipeline for analyzing and processing Brats-PED dataset.

### Key features and Usage
1. Preprocessing: Use `data_preprocessing.py` to convert raw NIfTI data into training-ready 2D slices.

2. Analysis: Run `statistics.py` to assess class balance (BG/FG ratios) or generate CSV-based dataset-wide statistical reports.

3. Visualization: Use the visualize_multimodal_slice function from `data_visualization.py` to inspect 4-channel multimodal MRI slices with segmentation overlays.

### Directory Structure
- `/data`: Stores raw input data, manifest files, and output statistics.

- `/eda`: Contains core processing and analysis scripts.

- `/visuals`: Contains visuals generated.

- `data_preprocessing.py`: Orchestrates the slice extraction and saving process.

- `data_visualization.py`: Functions for multi-modal slice rendering.

- `statistics.py`: Computes dataset-level metrics.