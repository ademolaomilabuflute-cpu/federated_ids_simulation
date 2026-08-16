"""
Central configuration for the federated simulation.

Values mirror the training setup described in the paper:
"A Lightweight Federated 1D-CNN for Real-Time, Privacy-Preserving
Intrusion Detection in Industrial IoT Edge Networks" (Sections 3.6, 4).
"""

# Data / model
N_FEATURES = 61          # Edge-IIoTset preprocessed feature count
N_CLASSES = 15           # 14 attack classes + normal traffic

# Federation
N_CLIENTS = 5
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Non-IID partitioning
DIRICHLET_ALPHA = 0.5    # lower alpha = more skewed / heterogeneous client shards

# Edge-hardware simulation (see paper Section 4 for the cgroups-enforced equivalent)
CLIENT_CPU_LIMIT = 4     # logical cores per client
CLIENT_RAM_LIMIT_GB = 4  # RAM per client

RANDOM_SEED = 42
