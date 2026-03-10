import numpy as np
import pandas as pd
import json
import os
import logging
from pathlib import Path
from .datacleaning import clean_dataframe
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse, save_npz
from typing import Optional
from .EDA import perform_eda, plot_correlation_heatmap, pca_reduction

logger = logging.getLogger(__name__)

def infer_task_type(y: pd.Series, classification_threshold=20, ratio_threshold=0.05):
    """Infer ML task type (classification or regression)."""
    y = y.dropna()
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"
    if np.issubdtype(y.dtype, np.number):
        n_unique = y.nunique()
        total = len(y)
        if n_unique <= classification_threshold or (n_unique / total) <= ratio_threshold:
            return "classification"
        else:
            return "regression"
    return "classification"


def process_features(
    cleaned_df: pd.DataFrame,
    target_col: str = "",
    problem_type: Optional[str] = None,
    save_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_pca: bool = True,
    pca_variance: float = 0.95,
    n_clusters: Optional[int] = None,
):
    """
    Process features for ML tasks and optionally save train/val arrays + metadata.

    Minimal return (per request): returns only X, y, and task_type.
    """
    encoders = {}

    # --- Handle target variable ---
    if target_col and target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    if target_col and target_col in cleaned_df.columns:
        y = cleaned_df[target_col]
        X_df = cleaned_df.drop(columns=[target_col]).copy()
        task_type = problem_type if problem_type in {"regression", "classification"} else infer_task_type(y)
    else:
        y = None
        X_df = cleaned_df.copy()
        task_type = problem_type if problem_type in {"kmeans_clustering"} else "kmeans_clustering"

    # --- Detect column types ---
    numeric_cols = X_df.select_dtypes(include=np.number).columns
    categorical_cols = []

    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        n_unique = X_df[col].nunique()
        if n_unique < 50:  # treat as categorical
            categorical_cols.append(col)
        else:
            # Drop text columns (high cardinality)
            logger.warning(
                f"Dropping column '{col}' with {n_unique} unique values (treated as text column). "
                "Text vectorization is not supported in this version."
            )
            X_df = X_df.drop(columns=[col])

    # --- Encode categorical features ---
    for col in categorical_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # --- Scale numeric features ---
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])
    else:
        scaler = None

    # --- Combine numeric + categorical ---
    X_final = X_df.to_numpy()
    feature_names = list(X_df.columns)

    # --- Encode target if classification ---
    if task_type == "classification" and y is not None:
        le_target = LabelEncoder()
        y_final = le_target.fit_transform(y.astype(str))
        encoders["target"] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
    elif task_type == "regression" and y is not None:
        y_final = y.astype(float).to_numpy()
    else:
        y_final = None
    
    if apply_pca:
        eda_result = perform_eda(X_final, corr_threshold=0.9, pca_variance=pca_variance)
        X_final = eda_result["X_reduced"]
        # PCA changes feature dimensions, so replace feature names with component names
        feature_names = [f"PC{i+1}" for i in range(X_final.shape[1])]
    else:
        eda_result = {"removed_corr_columns": [], "pca_model": None}
    # --- Optionally save train/val arrays + metadata ---

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if task_type in {"regression", "classification"}:
            X_train, X_val, y_train, y_val = train_test_split(
                X_final, y_final, test_size=test_size, random_state=random_state
            )

            if issparse(X_train):
                save_npz(save_path / "X_train.npz", X_train)
                save_npz(save_path / "X_val.npz", X_val)
                x_train_shape = X_train.shape
                x_val_shape = X_val.shape
                x_format = "sparse_npz"
            else:
                np.save(save_path / "X_train.npy", X_train)
                np.save(save_path / "X_val.npy", X_val)
                x_train_shape = X_train.shape
                x_val_shape = X_val.shape
                x_format = "dense_npy"

            np.save(save_path / "y_train.npy", y_train)
            np.save(save_path / "y_val.npy", y_val)

            metadata = {
                "problem_type": task_type,
                "target": target_col if target_col else None,
                "n_features": int(x_train_shape[1]) if len(x_train_shape) > 1 else 1,
                "train_samples": int(x_train_shape[0]),
                "val_samples": int(x_val_shape[0]),
                "feature_matrix_format": x_format,
                "categorical_cols": list(categorical_cols),
                "text_cols": [],
                "numeric_cols": list(numeric_cols),
                "feature_names": feature_names,
                "pca_applied": bool(apply_pca),
                "encoders": list(encoders.keys()),
                "vectorizers": [],
                "scaler_present": scaler is not None,
                "notes": "Encoders are kept in memory; no vectorizers are used."
            }

            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"✅ Saved train/val arrays and metadata to: {save_path}")

        else:
            if issparse(X_final):
                save_npz(save_path / "X_train.npz", X_final)
                x_shape = X_final.shape
                x_format = "sparse_npz"
            else:
                np.save(save_path / "X_train.npy", X_final)
                x_shape = X_final.shape
                x_format = "dense_npy"

            metadata = {
                "problem_type": task_type,
                "n_features": int(x_shape[1]) if len(x_shape) > 1 else 1,
                "train_samples": int(x_shape[0]),
                "feature_matrix_format": x_format,
                "feature_names": feature_names,
                "pca_applied": bool(apply_pca),
                "n_clusters": int(n_clusters) if n_clusters is not None else 3,
                "notes": "K-means clustering mode: feature matrix saved without target variable."
            }

            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"✅ Saved k-means clustering train matrix and metadata to: {save_path}")

    # --- Minimal return as requested ---
    return {
        "X": X_final,
        "y": y_final,
        "task_type": task_type
    }

# Refer here to run preprocessing on all files in a directory
# if __name__ == "__main__":

#     for files in os.listdir('main/raw_data'):
#         if files.endswith('.csv'):
#             df = pd.read_csv(os.path.join('main/raw_data',files))
#             df= clean_dataframe(df)
#             result = process_features(df, target_col="Sales ($)", save_dir="./processed_data")
#         if files.endswith('.xlsx'):
#             df = pd.read_excel(os.path.join('main/raw_data',files))
#             df= clean_dataframe(df)
#             result = process_features(df, target_col="Sales ($)", save_dir="./processed_data")
    
    
    

