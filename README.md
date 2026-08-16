# Federated 1D-CNN for IIoT Intrusion Detection — Simulation Code

Reference implementation accompanying the paper *"A Lightweight Federated
1D-CNN for Real-Time, Privacy-Preserving Intrusion Detection in Industrial
IoT Edge Networks."*

## What this is

A federated learning simulation built with [Flower](https://flower.ai) and
PyTorch, implementing the 60,623-parameter 1D-CNN and FedAvg training
protocol described in Sections 3–4 of the paper, evaluated on the
[Edge-IIoTset](https://ieeexplore.ieee.org/document/9751703) dataset under
non-IID Dirichlet partitioning (α = 0.5) across 5 simulated edge clients.

## Repository structure

| File               | Purpose                                                              |
|---------------------|-----------------------------------------------------------------------|
| `config.py`          | Central hyperparameters (Section 3.6)                                 |
| `model.py`            | 1D-CNN architecture, 60,623 parameters (Table 2)                      |
| `preprocessing.py`     | Cleaning, one-hot encoding, min-max scaling, class weighting (3.3–3.4) |
| `partition.py`          | Dirichlet non-IID client partitioning (Section 4)                     |
| `client.py`               | Flower client — local training/evaluation on one simulated gateway    |
| `server.py`                 | FedAvg strategy that retains the final trained global weights          |
| `metrics.py`                  | Per-class precision/recall/F1 evaluation (Table 4)                    |
| `simulate.py`                   | Entry point — runs training, then evaluates the actual trained model  |
| `ablation.py`                     | Sweeps client count and Dirichlet alpha (Table 6, Section 5.5)        |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Full run on real data
python simulate.py --data-path /path/to/EdgeIIoT_preprocessed.csv

# Pipeline smoke test only, on random synthetic data — NEVER report these numbers
python simulate.py --synthetic --num-rounds 2

# Client-count / heterogeneity ablation sweep (Table 6)
python ablation.py --data-path /path/to/EdgeIIoT_preprocessed.csv
```

Download Edge-IIoTset from its [official source](https://ieeexplore.ieee.org/document/9751703)
and adjust the categorical column names in `simulate.py::load_real_data`
if your copy's column headers differ. Missing or unreachable real data is
never silently substituted with synthetic data — `--data-path` is required
unless `--synthetic` is passed explicitly.

Each run writes `loss_history.csv` (real per-round evaluation loss) and
`per_class_report.csv` (per-class precision/recall/F1/support) to the
output directory, so reported figures and plots are regenerable from a
real run rather than transcribed by hand.

## On reproducibility

This code is a faithful implementation of the architecture and protocol
described in the paper. Exact reported metrics (95.97% F1, 8.4 ms
inference, 412 MB peak RAM, etc.) were measured in the original
experimental run under the hardware constraints described in Section 4
(4 CPU cores / 4 GB RAM per client). Federated deep learning involves
inherent stochasticity, so a fresh run — even with the same seed — may
land close to, rather than bit-identical to, the published numbers. This
is standard for research code of this kind.

## Citation

If you use this code, please cite the paper (full citation to be added
upon publication).
