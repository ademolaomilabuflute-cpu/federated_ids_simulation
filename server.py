"""
Server-side aggregation strategy: standard FedAvg (McMahan et al., 2017).

Sample-count-weighted averaging of client updates each round. No adaptive
optimizer is used, matching the paper's baseline configuration — see
Section 5.5 for the discussion of FedProx/SCAFFOLD as future work under
extreme heterogeneity.
"""
import flwr as fl


def build_strategy(
    fraction_fit: float = 1.0,
    min_fit_clients: int = 5,
    min_available_clients: int = 5,
) -> fl.server.strategy.FedAvg:
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )
