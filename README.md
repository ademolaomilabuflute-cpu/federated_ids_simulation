# Federated 1D-CNN for IIoT Intrusion Detection — Simulation Code

Reference implementation accompanying the paper *"A Lightweight Federated
1D-CNN for Real-Time, Privacy-Preserving Intrusion Detection in Industrial
IoT Edge Networks."*

## What this actually is

**This code produced every real number reported in the paper.** It is a
**single-process simulation**, not a distributed system: each client's local
training step and the server's weight-averaging step are executed in
sequence, in one Python process, on Kaggle's hosted notebook environment.
There is **no Flower, no gRPC, no network transport of any kind, and no
enforced per-client resource limits (cgroups or otherwise)**. The paper's
Section 3.1 describes a physically distributed, gRPC-based star topology as
the *target production design* — that system has not been built. Section 4
of the paper states this explicitly, and this repository's code matches
that description, not a distributed one.

An earlier version of this repository described a Flower/gRPC-based
implementation with cgroups-enforced 4-core/4GB client limits, and cited a
95.97% F1 / 8.4 ms / 412 MB result. **That description and those numbers
were never produced by any real experiment.** They have been fully replaced
here. If you are seeing a cached or forked copy with that content, treat it
as superseded by this version.

## The real, reported result

Produced by `Federated_IDS_EdgeIIoTset_Kaggle.ipynb`, run on Kaggle
(Python 3.12, PyTorch 2.10), on the full Edge-IIoTset dataset
(1,983,228 rows after preprocessing, 97 features, 15 classes), 5 simulated
clients, Dirichlet non-IID partitioning (α = 0.5), 20 communication rounds:

| Metric | Value |
|---|---|
| Model | 1D-CNN, **97,487 parameters** |
| Weighted F1 | 77.50% |
| Accuracy | 79.21% |
| Inference latency | 0.034 ms/flow |
| Peak RAM (whole process) | 5,820 MB |
| Communication overhead | 78 MB total over 20 rounds (93.59% less than shipping the 1,217 MB raw dataset once) |

Detection is markedly uneven across classes (7 of 14 attack types are not
detected at all under this configuration), and the training loss does not
converge within 20 rounds. See the paper's Section 5 for the full,
honest account, including the finding that a local-only baseline with no
aggregation outperforms both this federated result and a centralized
baseline (Table 3), and that a different backbone/strategy combination
(MLP + FedProx) outperforms this configuration in a separate comparison
sweep (Table 6).

## Repository structure

| File | Purpose |
|---|---|
| `Federated_IDS_EdgeIIoTset_Kaggle.ipynb` | The actual notebook that produced every real number in the paper. Self-contained: preprocessing, main federated run, baselines, comparison grid, ablations, and export, gated by toggles (`RUN_MAIN`, `RUN_BASELINES`, `RUN_COMPARISON`, `RUN_ABLATION`) documented in its own first cell. |

Earlier modular files (`client.py`, `server.py`, `model.py`, etc., built
around Flower) were a planned reference implementation for the target
distributed architecture. They were never run to produce any number in the
paper and have been removed from this repository to avoid the exact
confusion this replaces: code and documentation that looked authoritative
but did not correspond to any real result.

## Running it

Open `Federated_IDS_EdgeIIoTset_Kaggle.ipynb` on Kaggle (or any Jupyter
environment with PyTorch, pandas, and scikit-learn installed), attach the
[Edge-IIoTset dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
(Mohamed Amine Ferrag), and run all cells. The notebook's own header cell
explains the toggles and gives realistic runtime expectations per
configuration (the full main run took 3.5–9 hours on CPU; the comparison
grid took ~6 hours on GPU T4). Every run writes `run_summary.json` plus the
relevant CSVs/PNG to `/kaggle/working`, so every number in the paper is
independently regenerable, not transcribed by hand.

## On reproducibility

Federated deep learning involves inherent stochasticity, and this
particular training configuration does not converge (see the paper's
Section 5.3) — two runs of the identical configuration have produced
weighted F1 scores of 77.50% and 79.53% respectively. A fresh run should be
expected to land in a similar range, not bit-identical to either published
number. This is disclosed plainly in the paper rather than hidden.

## Citation

If you use this code, please cite the paper (full citation to be added
upon publication).
