import torch
import logging
from tqdm import tqdm
from ..modules.metrics import eval_result, compute_calibration_metrics
from ..datasets.noise_injection import PERTURBATION_REGISTRY

logger = logging.getLogger(__name__)


def robust_evaluate(model, dataloader, re_dict, args, perturbation_names=None):
    """鲁棒性评测入口

    对每种扰动运行评测，输出 F1/Acc + 与 clean 的 delta + 校准指标。

    Args:
        model: 训练好的模型
        dataloader: 测试数据 DataLoader
        re_dict: 关系字典
        args: 参数
        perturbation_names: 要评测的扰动名称列表，None 则评测全部
    Returns:
        results: dict, {perturbation_name: {f1, acc, delta_f1, ece, brier, nll}}
    """
    if perturbation_names is None:
        perturbation_names = list(PERTURBATION_REGISTRY.keys())

    # 先跑 clean baseline
    logger.info("=== Clean Evaluation ===")
    clean_result = _evaluate_single(model, dataloader, re_dict, args, perturb_fn=None)
    clean_f1 = clean_result['micro_f1']
    logger.info("Clean F1: {:.4f}, Acc: {:.4f}".format(clean_f1, clean_result['acc']))
    logger.info("Clean Calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        clean_result['ece'], clean_result['brier'], clean_result['nll']))

    results = {'clean': clean_result}

    # 逐扰动评测
    for name in perturbation_names:
        if name not in PERTURBATION_REGISTRY:
            logger.warning("Unknown perturbation: {}".format(name))
            continue
        logger.info("=== Perturbation: {} ===".format(name))
        perturb_fn = PERTURBATION_REGISTRY[name]
        res = _evaluate_single(model, dataloader, re_dict, args, perturb_fn=perturb_fn)
        res['delta_f1'] = res['micro_f1'] - clean_f1
        results[name] = res
        logger.info("{}: F1={:.4f} (delta={:.4f}), Acc={:.4f}, ECE={:.4f}".format(
            name, res['micro_f1'], res['delta_f1'], res['acc'], res['ece']))

    # 汇总
    logger.info("\n=== Robustness Summary ===")
    logger.info("{:<25s} {:>8s} {:>10s} {:>8s} {:>8s}".format(
        "Perturbation", "F1", "Delta_F1", "Acc", "ECE"))
    for name, res in results.items():
        delta = res.get('delta_f1', 0.0)
        logger.info("{:<25s} {:>8.4f} {:>10.4f} {:>8.4f} {:>8.4f}".format(
            name, res['micro_f1'], delta, res['acc'], res['ece']))

    return results


def _evaluate_single(model, dataloader, re_dict, args, perturb_fn=None):
    """单次评测（可选扰动）"""
    model.eval()
    true_labels, pred_labels = [], []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = (tup.to(args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
            params, labels = batch

            # 应用扰动
            if perturb_fn is not None:
                params = perturb_fn(params)

            params['mode'] = 'eval'
            outputs = model(**params)
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) >= 2 else outputs[0]
            else:
                logits = outputs

            preds = logits.argmax(-1)
            true_labels.extend(labels.view(-1).detach().cpu().tolist())
            pred_labels.extend(preds.view(-1).detach().cpu().tolist())
            all_logits.append(logits.detach().cpu())

    result = eval_result(true_labels, pred_labels, re_dict, logger)

    # 校准指标
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.tensor(true_labels)
    calib = compute_calibration_metrics(all_logits_t, all_labels_t)
    result.update(calib)

    return result
