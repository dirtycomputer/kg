import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    """计算 Acc / Micro-P / Micro-R / Micro-F1（复用 REMOTE 逻辑）"""
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            neg = name if use_name else rel2id[name]
            break

    for i in range(total):
        golden = true_labels[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1

    acc = float(correct) / float(total) if total > 0 else 0
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except ZeroDivisionError:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except ZeroDivisionError:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except ZeroDivisionError:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result


def compute_ece(logits, labels, n_bins=15):
    """Expected Calibration Error"""
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences.cpu().numpy() > bin_boundaries[i]) & (confidences.cpu().numpy() <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences.cpu().numpy()[in_bin].mean()
            avg_accuracy = accuracies.cpu().numpy()[in_bin].astype(float).mean()
            ece += abs(avg_confidence - avg_accuracy) * prop_in_bin
    return float(ece)


def compute_brier(logits, labels):
    """Brier Score"""
    probs = F.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    one_hot = F.one_hot(labels, num_classes).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return brier.item()


def compute_nll(logits, labels):
    """Negative Log-Likelihood"""
    return F.cross_entropy(logits, labels).item()


def compute_calibration_metrics(logits, labels, n_bins=15):
    """计算所有校准指标"""
    return {
        'ece': compute_ece(logits, labels, n_bins),
        'brier': compute_brier(logits, labels),
        'nll': compute_nll(logits, labels),
    }
