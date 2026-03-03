import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import cv2  # For fast resizing

def run_umap_analysis(output_root, n_samples=500):
    # 1. Load the manifest created during preprocessing
    manifest_path = os.path.join(output_root, "manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # 2. Sample data to fit in RAM (Balanced sample)
    if len(df) > n_samples:
        df_sample = df.groupby('is_tumor').sample(n_samples // 2, random_state=42)
    else:
        df_sample = df

    features = []
    labels = []

    print(f"Loading and resizing {len(df_sample)} slices...")
    for _, row in df_sample.iterrows():
        # Load the 4-channel image slice
        img_path = os.path.join(output_root, row['img_path'])
        img = np.load(img_path) # Shape: (240, 240, 4)
        
        # Resize to reduce dimensionality for UMAP performance
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Flatten (64*64*4 = 16384 features)
        features.append(img_resized.flatten())
        labels.append(row['is_tumor'])

    # 3. Scale the features
    print("Standardizing features...")
    X = np.array(features)
    X_scaled = StandardScaler().fit_transform(X)

    # 4. Run UMAP
    print("Running UMAP dimensionality reduction...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # 5. Visualize
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Is Tumor (1=Yes, 0=No)')
    plt.title('UMAP Projection of BraTS-PED 2D Slices')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


if __name__ == "__main__":
    OUTPUT_ROOT = r"D:\ECE324H1\preprocessed_dataset"
    run_umap_analysis(OUTPUT_ROOT)