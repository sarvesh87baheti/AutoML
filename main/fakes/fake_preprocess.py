import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def fake_preprocess(csv_path: str, output_dir: str, target_column: str = None):
    """
    Fake preprocessing script for testing AutoML regression pipeline.
    Works only for fully numeric datasets.
    
    - Loads CSV
    - Splits into train/val
    - Saves X_train.npy, y_train.npy, X_val.npy, y_val.npy
    - Creates metadata.json
    
    This is only for testing and does NOT replace real preprocessing.
    """

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine target column
    if target_column is None:
        target_column = df.columns[-1]   # assume last column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save numpy arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)

    # Metadata file
    metadata = {
        "problem_type": "classification",
        "target": target_column,
        "n_features": X_train.shape[1],
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0]
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Fake preprocessing completed.\nSaved to: {output_dir}")
