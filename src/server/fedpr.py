from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer
from src.client.fedpr import FedPRClient


class FedPRServer(FedAvgServer):
    algorithm_name = "FedPR"
    client_cls = FedPRClient

    def __init__(self, args: DictConfig, init_trainer: bool = True, init_model: bool = True, **kwargs):
        super().__init__(args, init_trainer=init_trainer, init_model=init_model)

