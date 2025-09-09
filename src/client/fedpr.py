import torch
from src.client.fedavg import FedAvgClient


class FedPRClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.projection_bases = {}
        self.energy_threshold = 0.8

    def set_parameters(self, package: dict):
        super().set_parameters(package)
        self._compute_projection_bases()

    def _compute_projection_bases(self):
        bases = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 只对参与聚合的参数进行计算且必须是可训练的
                if name not in self.regular_params_name or not param.requires_grad:
                    bases[name] = None
                    continue

                v = param.detach()
                # reshape：保持最后一维为 D
                if v.ndim == 0:
                    P = v.reshape(1, 1)
                elif v.ndim == 1:
                    P = v.unsqueeze(0)
                else:
                    P = v.reshape(-1, v.shape[-1])

                device = param.device
                # 协方差矩阵
                C = (P.t() @ P).cpu()
                U, S, Vh = torch.linalg.svd(C, full_matrices=False)

                # 找到最小 k 使得前 k 个奇异值占比 >= energy_threshold
                total = float(S.sum().item()) if S.numel() > 0 else 0.0
                cum = 0.0
                k = 0
                if total > 0.0:
                    for i in range(S.shape[0]):
                        cum += float(S[i].item())
                        if cum / total >= self.energy_threshold:
                            k = i + 1
                            break

                # 得到零空间
                U_null = U[:, k:].to(device=device).to(dtype=torch.float32).contiguous()
                bases[name] = U_null

        self.projection_bases = bases

    def _project_grad(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        U_null = self.projection_bases.get(name)
        if U_null is None:
            return grad

        U_null = U_null.to(grad.device).to(dtype=grad.dtype)

        grad_proj = (grad @ U_null) @ U_null.t()
        return grad_proj

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            self.optimizer.zero_grad()
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()

            # epoch 内所有 batch 的梯度已累积在 param.grad 中，逐参数投影
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if name in self.projection_bases and self.projection_bases[name] is not None:
                    param.grad.data.copy_(self._project_grad(name, param.grad.data))

            # 用投影后的梯度做一步更新
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
