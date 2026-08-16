"""
1D-CNN architecture for edge-constrained federated intrusion detection.

Matches Table 2 of the paper exactly:

    Input (61,1) -> Conv1D(32, k=3) -> MaxPool(2) -> Conv1D(64, k=3)
                  -> MaxPool(2) -> Flatten -> Dense(64) -> Dense(15, softmax)

Total parameters: 60,623 (verified in __main__ below).
"""
import torch
import torch.nn as nn


class FederatedIDSNet(nn.Module):
    def __init__(self, n_features: int = 61, n_classes: int = 15):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        flat_size = self._flattened_size(n_features)
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def _flattened_size(self, n_features: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, n_features)
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:            # (batch, 61) -> (batch, 1, 61)
            x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)          # raw logits — use with CrossEntropyLoss


if __name__ == "__main__":
    model = FederatedIDSNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")   # should print 60,623
    out = model(torch.randn(4, 61))
    print("Output shape:", tuple(out.shape))   # (4, 15)
