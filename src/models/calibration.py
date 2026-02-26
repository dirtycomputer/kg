import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize


class TemperatureScaling(nn.Module):
    """训练后温度缩放校准

    单参数 T，logits / T，在 val set 上用 L-BFGS 优化 T（最小化 NLL）。
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """在验证集上拟合温度参数

        Args:
            logits: [N, num_classes] 模型输出 logits
            labels: [N] 真实标签
            lr: 学习率
            max_iter: 最大迭代次数
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


class CalibrationMetrics:
    """校准评估指标: ECE, Brier Score, NLL"""

    @staticmethod
    def ece(probs, labels, n_bins=15):
        """Expected Calibration Error

        Args:
            probs: [N, num_classes] 预测概率
            labels: [N] 真实标签
            n_bins: bin 数量
        Returns:
            ece: float
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].float().mean()
                ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin

        return ece.item()

    @staticmethod
    def brier_score(probs, labels):
        """Brier Score

        Args:
            probs: [N, num_classes] 预测概率
            labels: [N] 真实标签
        Returns:
            brier: float
        """
        num_classes = probs.shape[1]
        one_hot = F.one_hot(labels, num_classes).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
        return brier.item()

    @staticmethod
    def nll(probs, labels):
        """Negative Log-Likelihood

        Args:
            probs: [N, num_classes] 预测概率
            labels: [N] 真实标签
        Returns:
            nll: float
        """
        log_probs = torch.log(probs + 1e-8)
        nll = F.nll_loss(log_probs, labels)
        return nll.item()

    @staticmethod
    def compute_all(logits, labels, n_bins=15):
        """计算所有校准指标

        Args:
            logits: [N, num_classes] 模型输出 logits
            labels: [N] 真实标签
        Returns:
            dict: {'ece': float, 'brier': float, 'nll': float}
        """
        probs = F.softmax(logits, dim=1)
        return {
            'ece': CalibrationMetrics.ece(probs, labels, n_bins),
            'brier': CalibrationMetrics.brier_score(probs, labels),
            'nll': CalibrationMetrics.nll(probs, labels),
        }
