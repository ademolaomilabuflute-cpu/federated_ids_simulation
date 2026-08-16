"""
Flower NumPyClient representing one simulated edge gateway.

Trains locally for LOCAL_EPOCHS on its own private data shard and returns
only weight updates to the server — raw traffic never leaves the client
(Section 3.1-3.2).
"""
from __future__ import annotations

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FederatedIDSNet


class IDSClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: FederatedIDSNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        device: torch.device,
        local_epochs: int = 5,
        lr: float = 1e-3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_total, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_total += self.criterion(logits, yb).item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                n += len(yb)
        return loss_total / n, n, {"accuracy": correct / n}
