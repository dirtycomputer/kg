import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceVIB(nn.Module):
    """序列级变分信息瓶颈

    在 encoder 输出上施加 VIB，对每个 token 位置进行信息压缩。
    使用固定 β_seq 系数。
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.logstd_encoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, mode='train'):
        """
        Args:
            hidden: [bsz, seq_len, hidden_size]
            mode: 'train' or 'eval'
        Returns:
            z: [bsz, seq_len, hidden_size] 重参数化后的表示
            kl_loss: scalar KL 散度
        """
        mu = self.mean_encoder(hidden)
        log_sigma = self.logstd_encoder(hidden)

        if mode == 'train':
            sigma = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(sigma)
            z = mu + eps * sigma
        else:
            z = mu

        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return z, kl_loss


class BetaNet(nn.Module):
    """自适应 β 网络

    根据相关性分数 r 和预测不确定性 u 动态计算 VIB 压缩强度 β。
    - r 低 + u 高 → β 大（强压缩去噪）
    - r 高 + u 低 → β 小（保留信息）
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, r, u):
        """
        Args:
            r: [bsz, 1] 相关性分数
            u: [bsz, 1] 归一化不确定性 (预测熵 / log(num_classes))
        Returns:
            beta: [bsz, 1] 自适应压缩系数, clamped to [1e-4, 1.0]
        """
        x = torch.cat([r, u], dim=-1)  # [bsz, 2]
        beta = self.net(x)
        beta = beta.clamp(1e-4, 1.0)
        return beta


class EntityVIB(nn.Module):
    """实体级自适应变分信息瓶颈

    在提取 head/tail 实体表示后施加 VIB，
    使用 BetaNet 根据相关性和不确定性自适应调节压缩强度。
    """

    def __init__(self, hidden_size=768, num_classes=None):
        super().__init__()
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.logstd_encoder = nn.Linear(hidden_size, hidden_size)
        self.beta_net = BetaNet()
        self.num_classes = num_classes

    def forward(self, entity_hidden, r, mode='train', preliminary_logits=None):
        """
        Args:
            entity_hidden: [bsz, hidden_size] 实体表示
            r: [bsz, 1] 相关性分数
            mode: 'train' or 'eval'
            preliminary_logits: [bsz, num_classes] 初步分类 logits（用于计算不确定性）
        Returns:
            z: [bsz, hidden_size]
            weighted_kl: scalar 加权 KL 损失
            beta: [bsz, 1] 自适应 β
        """
        mu = self.mean_encoder(entity_hidden)
        log_sigma = self.logstd_encoder(entity_hidden)

        if mode == 'train':
            sigma = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(sigma)
            z = mu + eps * sigma
        else:
            z = mu

        # Per-sample KL
        kl_per_sample = -0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp()).mean(dim=-1, keepdim=True)  # [bsz, 1]

        # 计算归一化不确定性 u
        if preliminary_logits is not None and self.num_classes is not None:
            probs = F.softmax(preliminary_logits.detach(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)  # [bsz, 1]
            u = entropy / (torch.log(torch.tensor(float(self.num_classes))) + 1e-8)  # 归一化到 [0, 1]
        else:
            u = torch.zeros_like(r)

        # 自适应 β
        beta = self.beta_net(r.detach(), u)  # [bsz, 1]
        weighted_kl = (beta * kl_per_sample).mean()

        return z, weighted_kl, beta
