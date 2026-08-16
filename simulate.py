"""
Entry point: runs the federated simulation described in the paper.

Usage:
    python simulate.py --data-path /path/to/EdgeIIoT_preprocessed.csv

Reproducibility note: this is a faithful reference implementation of the
architecture and protocol described in Sections 3-4 of the paper. Exact
reported metrics (F1, latency, peak RAM) came from the original run under
the cgroups-enforced hardware limits in Section 4, and may vary slightly
run-to-run — expected behavior for federated deep-learning experiments,
not a discrepancy in the method itself.
"""
from __future__ import annotations

import argparse

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import IDSClient
from config import (
    BATCH_SIZE,
    DIRICHLET_ALPHA,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    N_CLASSES,
    N_CLIENTS,
    N_FEATURES,
    NUM_ROUNDS,
    RANDOM_SEED,
)
from model import FederatedIDSNet
from partition import dirichlet_partition
from preprocessing import clean_dataframe, compute_class_weights, encode_categoricals, scale_features
from server import build_strategy


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    df = clean_dataframe(df)

    # Adjust these to match Edge-IIoTset's actual column headers in your copy.
    categorical_cols = [c for c in ["proto", "http.request.method"] if c in df.columns]
    if categorical_cols:
        df = encode_categoricals(df, categorical_cols)

    label_col = "Attack_label" if "Attack_label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]
    return df, feature_cols, label_col


def make_client_fn(shards, feature_cols, label_col, device):
    def client_fn(cid: str) -> IDSClient:
        shard = shards[int(cid)]
        scaled, _ = scale_features(shard, feature_cols)
        X = torch.tensor(scaled[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(scaled[label_col].values, dtype=torch.long)

        n_val = max(1, int(0.2 * len(X)))
        train_ds = TensorDataset(X[n_val:], y[n_val:])
        val_ds = TensorDataset(X[:n_val], y[:n_val])

        class_weights = torch.tensor(compute_class_weights(y[n_val:].numpy(), N_CLASSES))
        model = FederatedIDSNet(n_features=N_FEATURES, n_classes=N_CLASSES)

        return IDSClient(
            model=model,
            train_loader=DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            val_loader=DataLoader(val_ds, batch_size=BATCH_SIZE),
            class_weights=class_weights,
            device=device,
            local_epochs=LOCAL_EPOCHS,
            lr=LEARNING_RATE,
        )

    return client_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to preprocessed Edge-IIoTset CSV")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")  # GPU disabled to mirror the edge-gateway constraint (Section 4)

    df, feature_cols, label_col = load_and_prepare(args.data_path)
    labels = df[label_col].values
    partitions = dirichlet_partition(labels, N_CLIENTS, DIRICHLET_ALPHA, seed=RANDOM_SEED)
    shards = [df.iloc[idx].reset_index(drop=True) for idx in partitions]

    fl.simulation.start_simulation(
        client_fn=make_client_fn(shards, feature_cols, label_col, device),
        num_clients=N_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=build_strategy(min_fit_clients=N_CLIENTS, min_available_clients=N_CLIENTS),
    )


if __name__ == "__main__":
    main()
