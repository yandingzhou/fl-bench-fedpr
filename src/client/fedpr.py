from src.client.fedavg import FedAvgClient
from collections import OrderedDict
from src.utils.constants import CONFIG
from src.utils.adam_svd import AdamSVD


class FedPRClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.projection_bases = {}
        self.energy_threshold = 0.8

        trainable_prompt = []
        for idx in range(CONFIG.FL.CLIENTS_NUM):
            m_param = [v for k, v in self.model.base.prompter.named_parameters() if v.requires_grad]
            trainable_prompt.append(m_param)
        self.optimizer = [AdamSVD(trainable_prompt[idx], lr=CONFIG.SOLVER.LR[idx], weight_decay=CONFIG.SOLVER.WEIGHT_DECAY, ratio=CONFIG.SOLVER.RATIO) for idx in range(CONFIG.FL.CLIENTS_NUM)]

    def set_parameters(self, package: dict):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if (
                package["optimizer_state"]
                and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer[self.client_id].load_state_dict(package["optimizer_state"])
        else:
            self.optimizer[self.client_id].load_state_dict(self.init_optimizer_state)

        self.optimizer[self.client_id].get_eigens(fea_in=package["fea_in"])
        self.optimizer[self.client_id].get_transforms()

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )