import torch
import logging
from tqdm import tqdm
from ..models.calibration import TemperatureScaling, CalibrationMetrics
from ..modules.metrics import compute_calibration_metrics

logger = logging.getLogger(__name__)


def calibration_evaluate(model, dataloader, args, temperature=None):
    """校准评测入口

    Args:
        model: 训练好的模型
        dataloader: 验证/测试数据 DataLoader
        args: 参数
        temperature: 已拟合的 TemperatureScaling 模块，None 则不做温度缩放
    Returns:
        results: dict, {before_calib: {ece, brier, nll}, after_calib: {ece, brier, nll}}
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits", leave=False):
            params, labels = batch
            params = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in params.items()}
            labels = labels.to(args.device)
            params['mode'] = 'eval'
            outputs = model(**params)
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) >= 2 else outputs[0]
            else:
                logits = outputs

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.view(-1).detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 校准前
    before = compute_calibration_metrics(all_logits, all_labels)
    logger.info("Before calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        before['ece'], before['brier'], before['nll']))

    results = {'before_calib': before}

    # 温度缩放
    if temperature is not None:
        scaled_logits = temperature(all_logits)
        after = compute_calibration_metrics(scaled_logits, all_labels)
        logger.info("After calibration (T={:.4f}): ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
            temperature.temperature.item(), after['ece'], after['brier'], after['nll']))
        results['after_calib'] = after
        results['temperature'] = temperature.temperature.item()

    return results


def fit_temperature(model, val_dataloader, args):
    """在验证集上拟合温度参数

    Args:
        model: 训练好的模型
        val_dataloader: 验证数据 DataLoader
        args: 参数
    Returns:
        temp_scaling: 拟合好的 TemperatureScaling 模块
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Collecting val logits", leave=False):
            params, labels = batch
            params = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in params.items()}
            labels = labels.to(args.device)
            params['mode'] = 'eval'
            outputs = model(**params)
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) >= 2 else outputs[0]
            else:
                logits = outputs

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.view(-1).detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    temp_scaling = TemperatureScaling()
    optimal_t = temp_scaling.fit(all_logits, all_labels)
    logger.info("Optimal temperature: {:.4f}".format(optimal_t))

    return temp_scaling
