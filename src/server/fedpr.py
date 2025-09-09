from omegaconf import DictConfig
import torch
from src.server.fedavg import FedAvgServer
from src.client.fedpr import FedPRClient
from copy import deepcopy
from collections import defaultdict


class FedPRServer(FedAvgServer):
    algorithm_name = "FedPR"
    client_cls = FedPRClient

    def __init__(self, args: DictConfig, init_trainer: bool = True, init_model: bool = True, **kwargs):
        super().__init__(args, init_trainer=init_trainer, init_model=init_model)

    def package(self, client_id: int):
        fea_in = defaultdict(dict)
        for idx, (k, p) in enumerate(self.model.enc.prompter.named_parameters()):
            fea_in[idx] = torch.bmm(p.transpose(1, 2), p)

        return dict(
            client_id=client_id,
            local_epoch=self.client_local_epoches[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.client_optimizer_states[client_id],
            lr_scheduler_state=self.client_lr_scheduler_states[client_id],
            return_diff=self.return_diff,
            fea_in=fea_in
        )