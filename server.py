"""
Server-side aggregation strategy: standard FedAvg (McMahan et al., 2017).

Sample-count-weighted averaging of client updates each round. No adaptive
optimizer is used, matching the paper's baseline configuration — see
Section 5.5 for the discussion of FedProx/SCAFFOLD as future work under
extreme heterogeneity.
"""
from __future__ import annotations

import flwr as fl


class SavingFedAvg(fl.server.strategy.FedAvg):
    """
    Plain FedAvg does not expose the final aggregated model once a
    simulation ends — Flower's History object only carries metrics, not
    weights. This subclass keeps a reference to the most recently
    aggregated parameters after every round, and per-round evaluation
    loss, so the driver script can (a) load the truly trained global
    model for final evaluation instead of a fresh random one, and
    (b) plot a real loss-vs-round curve.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: fl.common.Parameters | None = None
        self.loss_history: list[tuple[int, float]] = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_loss is not None:
            self.loss_history.append((server_round, aggregated_loss))
        return aggregated_loss, metrics


def build_strategy(
    fraction_fit: float = 1.0,
    min_fit_clients: int = 5,
    min_available_clients: int = 5,
) -> SavingFedAvg:
    return SavingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )
