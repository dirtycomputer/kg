import torch
import torch.nn as nn
import torch.nn.functional as F


class RelevanceGate(nn.Module):
    """相关性感知门控模块

    通过文本和视觉表示计算图文相关性分数 r ∈ [0,1]，
    用于缩放视觉分支输出，抑制不相关图像的噪声。
    """

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        # gate_head: [t_pool; v_pool; t_pool * v_pool] -> r
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, t_hidden, v_hidden):
        """
        Args:
            t_hidden: [bsz, seq_len, hidden_size] 文本 encoder 输出
            v_hidden: [bsz, 12, hidden_size] 视觉 encoder 输出
        Returns:
            r: [bsz, 1] 相关性分数
            gate_entropy_loss: scalar 熵正则损失
        """
        # Pool
        t_pool = self.text_proj(t_hidden[:, 0, :])       # [bsz, hidden]
        v_pool = self.vision_proj(v_hidden.mean(dim=1))   # [bsz, hidden]

        # Gate
        gate_input = torch.cat([t_pool, v_pool, t_pool * v_pool], dim=-1)
        r = self.gate_head(gate_input)  # [bsz, 1]

        # 熵正则: 防止 r 退化到 0 或 1
        # H(r) = -r*log(r) - (1-r)*log(1-r)
        eps = 1e-7
        entropy = -(r * torch.log(r + eps) + (1 - r) * torch.log(1 - r + eps))
        gate_entropy_loss = -entropy.mean()  # 最大化熵 → 最小化负熵

        return r, gate_entropy_loss


class GateContrastiveLoss(nn.Module):
    """门控对比损失

    自监督训练信号：
    - 正样本: 原始 (text, image) 对 → target=1
    - 负样本: batch 内 roll 图像 → target=0
    """

    def __init__(self, neg_num=1):
        super().__init__()
        self.neg_num = neg_num

    def forward(self, gate_module, t_hidden, v_hidden):
        """
        Args:
            gate_module: RelevanceGate 实例
            t_hidden: [bsz, seq_len, hidden_size]
            v_hidden: [bsz, 12, hidden_size]
        Returns:
            loss: scalar 对比损失
        """
        # 正样本
        r_pos, _ = gate_module(t_hidden, v_hidden)
        loss = F.binary_cross_entropy(r_pos, torch.ones_like(r_pos))

        # 负样本: batch 内 roll
        for s in range(1, self.neg_num + 1):
            v_neg = v_hidden.roll(shifts=s, dims=0)
            r_neg, _ = gate_module(t_hidden, v_neg)
            loss = loss + F.binary_cross_entropy(r_neg, torch.zeros_like(r_neg))

        return loss
