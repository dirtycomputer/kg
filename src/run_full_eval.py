"""
完整评测脚本: Clean + Robustness + Calibration
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
import json
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unimo_model import RelevanceAwareUnimoREModel
from datasets.umke_dataset import UMKEProcessor, UMKEDataset
from datasets.noise_injection import PERTURBATION_REGISTRY
from models.calibration import TemperatureScaling
from modules.metrics import eval_result, compute_calibration_metrics

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../eval_results.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def evaluate_single(model, dataloader, re_dict, device, perturb_fn=None):
    model.eval()
    true_labels, pred_labels = [], []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            params, labels = batch
            params = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in params.items()}
            labels = labels.to(device)

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
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.tensor(true_labels)
    calib = compute_calibration_metrics(all_logits_t, all_labels_t)
    result.update(calib)
    return result, all_logits_t, all_labels_t


def main():
    set_seed(1)

    data_dir = '../data/UMRE'
    ckpt_path = '../checkpoints/UMRE_16_2e-05_gate_vib_full_20260226_104649/best_model.pth'

    args = argparse.Namespace(
        max_seq=128, use_box=True, use_cap=True, use_dep=False,
        use_gate=True, use_vib=True, use_seq_vib=True, use_entity_vib=True,
        beta_seq=0.01, alpha_gate=0.5, lambda_entropy=0.1,
        bert_name='../model/bert-base-uncased', vit_name='../model/clip-vit-base-patch32',
        device='cuda',
    )

    data_path = {
        'train': os.path.join(data_dir, 'train_set.json'),
        'valid': os.path.join(data_dir, 'val_set.json'),
        'test': os.path.join(data_dir, 'test_set.json'),
        'train_ent_dict': os.path.join(data_dir, 'pos_umke.json'),
        'valid_ent_dict': os.path.join(data_dir, 'pos_umke.json'),
        'test_ent_dict': os.path.join(data_dir, 'pos_umke.json'),
    }
    re_path = os.path.join(data_dir, 'rel2id.json')
    img_path = {m: os.path.join(data_dir, 'UMRE_IMG') for m in ['train', 'valid', 'test']}
    cap_path = {m: os.path.join(data_dir, 'cap_qwenvl_simply.json') for m in ['train', 'valid', 'test']}

    logger.info("Loading data...")
    processor = UMKEProcessor(data_path, re_path, args.bert_name, args.vit_name)
    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer

    test_dataset = UMKEDataset(processor, img_path, None, cap_path, args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    valid_dataset = UMKEDataset(processor, img_path, None, cap_path, args, mode='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading model from %s", ckpt_path)
    model = RelevanceAwareUnimoREModel(num_labels, tokenizer, args)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to('cuda')
    model.eval()

    all_results = {}

    # ============================================================
    # Part 1: Clean Evaluation
    # ============================================================
    logger.info("=" * 60)
    logger.info("PART 1: CLEAN EVALUATION")
    logger.info("=" * 60)
    clean_result, test_logits, test_labels = evaluate_single(model, test_loader, re_dict, 'cuda')
    all_results['clean'] = clean_result
    logger.info("Clean Test: Acc={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(
        clean_result['acc'], clean_result['micro_p'], clean_result['micro_r'], clean_result['micro_f1']))
    logger.info("Clean Calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        clean_result['ece'], clean_result['brier'], clean_result['nll']))

    # ============================================================
    # Part 2: Robustness Evaluation
    # ============================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: ROBUSTNESS EVALUATION")
    logger.info("=" * 60)

    clean_f1 = clean_result['micro_f1']
    for perturb_name, perturb_fn in PERTURBATION_REGISTRY.items():
        logger.info("--- Perturbation: %s ---", perturb_name)
        res, _, _ = evaluate_single(model, test_loader, re_dict, 'cuda', perturb_fn=perturb_fn)
        res['delta_f1'] = res['micro_f1'] - clean_f1
        all_results[perturb_name] = res
        logger.info("{}: F1={:.4f} (delta={:+.4f}), Acc={:.4f}, ECE={:.4f}".format(
            perturb_name, res['micro_f1'], res['delta_f1'], res['acc'], res['ece']))

    # ============================================================
    # Part 3: Calibration (Temperature Scaling)
    # ============================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 3: CALIBRATION (TEMPERATURE SCALING)")
    logger.info("=" * 60)

    logger.info("Collecting validation logits for temperature fitting...")
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in valid_loader:
            params, labels = batch
            params = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in params.items()}
            labels = labels.to('cuda')
            params['mode'] = 'eval'
            outputs = model(**params)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            val_logits_list.append(logits.detach().cpu())
            val_labels_list.append(labels.view(-1).detach().cpu())

    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    ts = TemperatureScaling()
    optimal_t = ts.fit(val_logits, val_labels)
    logger.info("Optimal temperature: {:.4f}".format(optimal_t))

    before_calib = compute_calibration_metrics(test_logits, test_labels)
    scaled_test_logits = ts(test_logits)
    after_calib = compute_calibration_metrics(scaled_test_logits, test_labels)

    all_results['calibration'] = {
        'temperature': optimal_t,
        'before': before_calib,
        'after': after_calib,
    }

    logger.info("Before calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        before_calib['ece'], before_calib['brier'], before_calib['nll']))
    logger.info("After  calibration: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        after_calib['ece'], after_calib['brier'], after_calib['nll']))

    # ============================================================
    # Summary Table
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("FULL RESULTS SUMMARY")
    logger.info("=" * 80)
    header = "{:<25s} {:>8s} {:>8s} {:>8s} {:>10s} {:>8s} {:>8s} {:>8s}".format(
        "Setting", "Acc", "P", "R", "F1", "ECE", "Brier", "NLL")
    logger.info(header)
    logger.info("-" * 80)

    for name in ['clean'] + list(PERTURBATION_REGISTRY.keys()):
        r = all_results[name]
        delta_str = ""
        if name != 'clean':
            delta_str = "({:+.4f})".format(r.get('delta_f1', 0))
        logger.info("{:<25s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:<8s} {:>6.4f} {:>8.4f} {:>8.4f}".format(
            name, r['acc'], r['micro_p'], r['micro_r'], r['micro_f1'], delta_str,
            r['ece'], r['brier'], r['nll']))

    logger.info("-" * 80)
    logger.info("Temperature Scaling: T={:.4f}".format(optimal_t))
    logger.info("  Before: ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        before_calib['ece'], before_calib['brier'], before_calib['nll']))
    logger.info("  After:  ECE={:.4f}, Brier={:.4f}, NLL={:.4f}".format(
        after_calib['ece'], after_calib['brier'], after_calib['nll']))
    logger.info("=" * 80)

    with open('../eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to eval_results.json and eval_results.log")


if __name__ == "__main__":
    main()
