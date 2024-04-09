import flwr as fl
from typing import List, Tuple, Dict

# 模型训练的聚合函数
# A function to aggregate the metrics from all clients in training
def fit_metrics_aggregation(data: List[Tuple[int, Dict]]) -> Dict:
    if not data:
        return {"accuracy": None, "loss": None}

    total_weight = sum(weight for weight, _ in data)
    if total_weight == 0:
        return {"accuracy": None, "loss": None}
    
    total_accuracy = sum(weight * metrics['accuracy'] for weight, metrics in data) / total_weight
    total_loss = sum(weight * metrics['loss'] for weight, metrics in data) / total_weight
    
    return {"accuracy": total_accuracy, "loss": total_loss}

config = fl.server.ServerConfig(num_rounds=20)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    fit_metrics_aggregation_fn=fit_metrics_aggregation,
)

fl.server.start_server(config=config, strategy=strategy)