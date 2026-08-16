"""
Local preprocessing pipeline run independently on each client (Section 3.3),
plus the class-weighting scheme used to handle imbalance (Section 3.4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and rows containing NaN/infinite values."""
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical fields (protocol type, HTTP method, etc.)."""
    cols_present = [c for c in categorical_cols if c in df.columns]
    return pd.get_dummies(df, columns=cols_present) if cols_present else df


def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Min-max scale numeric features to [0, 1].
    Extrema are computed on THIS client's local data only (Section 3.3) —
    no statistics are shared across clients.
    """
    df = df.copy()
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Per-client class weighting: W_c = N / (C * n_c)   (Section 3.4)

    Penalizes misclassification of rare attack classes more heavily than
    misclassification of the majority (benign) class, without the memory
    cost of synthetic oversampling (e.g. SMOTE).
    """
    n = len(labels)
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = np.sum(labels == c)
        weights[c] = n / (n_classes * n_c) if n_c > 0 else 0.0
    return weights
