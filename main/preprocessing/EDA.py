import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.sparse import issparse

def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.9):
    """
    Plots correlation heatmap and returns columns to drop based on threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return drop_cols

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9):
    """
    Automatically remove columns with correlation above threshold.
    Returns filtered DataFrame and list of removed columns.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    df_filtered = df.drop(columns=drop_cols)
    return df_filtered, drop_cols

def pca_reduction(X, variance_threshold: float = 0.95):
    """
    Reduces features using PCA while retaining specified variance.
    Works with dense NumPy arrays or sparse matrices.
    """
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    pca = PCA(n_components=variance_threshold)
    X_reduced = pca.fit_transform(X_dense)
    return X_reduced, pca

def perform_eda(X, target_col: str = "", corr_threshold: float = 0.9,
                pca_variance: float = 0.95, plot_corr: bool = True):
    """
    Perform EDA and PCA:
      - If input is DataFrame:
          • Optionally plot correlation heatmap
          • Remove highly correlated features automatically
      - If input is NumPy array / sparse, skip correlation heatmap
      - Apply PCA for dimensionality reduction
    Returns:
      dict containing:
        • X_original: array before PCA
        • X_reduced: array after PCA
        • pca_model: fitted PCA object
        • removed_corr_columns: dropped columns due to correlation (names or info)
    """
    removed_cols = []

    if isinstance(X, pd.DataFrame):
        numeric_df = X.select_dtypes(include=np.number)
        if plot_corr:
            _ = plot_correlation_heatmap(numeric_df, threshold=corr_threshold)
        X_filtered, removed_cols = remove_highly_correlated_features(numeric_df, threshold=corr_threshold)
        X_numeric = X_filtered.to_numpy()
    else:
        # X is already numeric (NumPy array or sparse)
        if issparse(X):
            X_numeric = X.toarray()
        else:
            X_numeric = X
        removed_cols = f"PCA will reduce dimensions from {X_numeric.shape[1]} features"

    # Apply PCA reduction
    X_reduced, pca_model = pca_reduction(X_numeric, variance_threshold=pca_variance)

    return {
        "X_original": X_numeric,
        "X_reduced": X_reduced,
        "pca_model": pca_model,
        "removed_corr_columns": removed_cols
    }