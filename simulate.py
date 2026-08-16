"""
Entry point: runs the federated simulation described in the paper, then
evaluates the ACTUAL trained global model (not a fresh random one) on a
held-out test set.

Usage:
    python simulate.py --data-path /path/to/EdgeIIoT_preprocessed.csv

    # Dry run with random synthetic data (for smoke-testing the pipeline
    # only — never use this to report real results):
    python simulate.py --synthetic

Fixes relative to earlier drafts of this script:
  - Final evaluation now loads the strategy's actual aggregated weights
    after training, instead of instantiating a fresh, untrained model.
  - Missing/absent real data no longer silently falls back to random
    synthetic data; that now requires the explicit --synthetic flag.
  - Reports a full per-class precision/recall/F1 breakdown, not just one
    aggregate weighted score.
  - Client-reported evaluation loss is true categorical cross-entropy,
    matched with a real per-round loss history saved to disk.

Reproducibility note: this is a faithful implementation of the
architecture and protocol described in Sections 3-4 of the paper. Results
depend on the real Edge-IIoTset data and hardware you run it on, and may
differ from previously published figures — treat this as the tool for
generating fresh, verifiable numbers, not as a guarantee of reproducing
old ones.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

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
from metrics import evaluate_global, print_report
from model import FederatedIDSNet
from partition import dirichlet_partition
from preprocessing import clean_dataframe, compute_class_weights, encode_categoricals, scale_features
from server import SavingFedAvg, build_strategy


def load_real_data(data_path: str):
    df = pd.read_csv(data_path)
    df = clean_dataframe(df)

    categorical_cols = [c for c in ["proto", "http.request.method"] if c in df.columns]
    if categorical_cols:
        df = encode_categoricals(df, categorical_cols)

    label_col = "Attack_label" if "Attack_label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_col].values
    class_names, y = np.unique(y_raw, return_inverse=True)
    return X, y, feature_cols, list(class_names)


def load_synthetic_data(n_samples: int = 15000, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
    print("[WARNING] Using RANDOM SYNTHETIC data — this is a pipeline smoke test only. "
          "Any metrics printed below are meaningless and must never be reported as results.",
          file=sys.stderr)
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    class_names = [f"class_{i}" for i in range(n_classes)]
    return X, y, [f"f{i}" for i in range(n_features)], class_names


def make_client_fn(shards, feature_cols, device):
    def client_fn(cid: str) -> IDSClient:
        X_c, y_c = shards[int(cid)]
        n_val = max(1, int(0.2 * len(X_c)))
        Xt = torch.tensor(X_c, dtype=torch.float32)
        yt = torch.tensor(y_c, dtype=torch.long)

        train_ds = TensorDataset(Xt[n_val:], yt[n_val:])
        val_ds = TensorDataset(Xt[:n_val], yt[:n_val])

        class_weights = torch.tensor(compute_class_weights(yt[n_val:].numpy(), N_CLASSES))
        model = FederatedIDSNet(n_features=len(feature_cols), n_classes=N_CLASSES)

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
    parser.add_argument("--data-path", help="Path to preprocessed Edge-IIoTset CSV")
    parser.add_argument("--synthetic", action="store_true",
                         help="Explicitly run a smoke test on random synthetic data instead of real data")
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--num-clients", type=int, default=N_CLIENTS)
    parser.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if not args.synthetic and not args.data_path:
        parser.error("Provide --data-path to a real dataset, or pass --synthetic to explicitly "
                      "run a smoke test on random data. Silently substituting synthetic data for "
                      "missing real data is not supported, to avoid ever mistaking placeholder "
                      "numbers for real results.")
    if not args.synthetic and not os.path.exists(args.data_path):
        parser.error(f"--data-path '{args.data_path}' does not exist. Pass --synthetic if you "
                      f"intended to run a pipeline smoke test instead.")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")  # GPU disabled to mirror the edge-gateway constraint (Section 4)

    if args.synthetic:
        X, y, feature_cols, class_names = load_synthetic_data()
    else:
        X, y, feature_cols, class_names = load_real_data(args.data_path)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    _, scaler = scale_features(train_df, feature_cols)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    partitions_idx = dirichlet_partition(y_train, args.num_clients, args.alpha, seed=RANDOM_SEED)
    shards = [(X_train_scaled[idx], y_train[idx]) for idx in partitions_idx]
    for i, (Xc, _) in enumerate(shards):
        print(f"[PARTITION] Client {i + 1}: {len(Xc)} samples allocated.")

    strategy = build_strategy(min_fit_clients=args.num_clients, min_available_clients=args.num_clients)
    assert isinstance(strategy, SavingFedAvg)

    print(f"\n[FEDERATION] Starting {args.num_rounds} global rounds across {args.num_clients} clients...\n")
    fl.simulation.start_simulation(
        client_fn=make_client_fn(shards, feature_cols, device),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Save the real per-round loss history (for a genuine Figure-2-style plot).
    loss_csv_path = os.path.join(args.out_dir, "loss_history.csv")
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "eval_loss"])
        writer.writerows(strategy.loss_history)
    print(f"\n[LOG] Per-round loss history saved to {loss_csv_path}")

    # --- Critical fix: load the model that was ACTUALLY trained, not a fresh one. ---
    if strategy.latest_parameters is None:
        print("[ERROR] No aggregated parameters were produced — federated training did not "
              "complete successfully. Final evaluation cannot proceed on an untrained model.",
              file=sys.stderr)
        sys.exit(1)

    trained_ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
    global_model = FederatedIDSNet(n_features=len(feature_cols), n_classes=N_CLASSES).to(device)
    state_dict = global_model.state_dict()
    for key, val in zip(state_dict.keys(), trained_ndarrays):
        state_dict[key] = torch.tensor(val)
    global_model.load_state_dict(state_dict, strict=True)

    test_ds = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 65)
    print("FINAL GLOBAL EVALUATION — trained aggregated model, held-out test set")
    print("=" * 65)
    aggregate, per_class = evaluate_global(global_model, test_loader, device, class_names)
    print_report(aggregate, per_class)

    report_csv_path = os.path.join(args.out_dir, "per_class_report.csv")
    with open(report_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class)
    print(f"\n[LOG] Per-class report saved to {report_csv_path}")


if __name__ == "__main__":
    main()
