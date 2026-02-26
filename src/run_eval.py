import os
import argparse
import logging
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.unimo_model import RelevanceAwareUnimoREModel
from datasets.umke_dataset import UMKEProcessor, UMKEDataset
from eval.robust_eval import robust_evaluate
from eval.calibration_eval import calibration_evaluate, fit_temperature
import warnings
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluation: clean + robust + calibration")

    # 模型
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str)

    # 数据
    parser.add_argument('--dataset_name', default='UMKE', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--img_dir', default=None, type=str)
    parser.add_argument('--dep_dir', default=None, type=str)
    parser.add_argument('--cap_path', default=None, type=str)
    parser.add_argument('--re_path', default=None, type=str)

    # 模型参数
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--load_path', required=True, type=str, help="Path to trained model checkpoint")

    # 特征开关
    parser.add_argument('--use_box', action='store_true')
    parser.add_argument('--use_cap', action='store_true')
    parser.add_argument('--use_dep', action='store_true')

    # 新模块开关
    parser.add_argument('--use_gate', action='store_true', default=True)
    parser.add_argument('--no_gate', action='store_false', dest='use_gate')
    parser.add_argument('--use_vib', action='store_true', default=True)
    parser.add_argument('--no_vib', action='store_false', dest='use_vib')
    parser.add_argument('--use_seq_vib', action='store_true', default=True)
    parser.add_argument('--no_seq_vib', action='store_false', dest='use_seq_vib')
    parser.add_argument('--use_entity_vib', action='store_true', default=True)
    parser.add_argument('--no_entity_vib', action='store_false', dest='use_entity_vib')

    # 超参数
    parser.add_argument('--beta_seq', default=0.01, type=float)
    parser.add_argument('--alpha_gate', default=0.5, type=float)
    parser.add_argument('--lambda_entropy', default=0.1, type=float)

    # 评测开关
    parser.add_argument('--do_robust_eval', action='store_true')
    parser.add_argument('--do_calibrate', action='store_true')
    parser.add_argument('--perturbations', nargs='+', default=None,
                        help="Specific perturbations to evaluate")

    args = parser.parse_args()

    if not args.use_vib:
        args.use_seq_vib = False
        args.use_entity_vib = False

    set_seed(args.seed)

    # 数据路径
    data_path = {
        'train': os.path.join(args.data_dir, 'train_set.json'),
        'valid': os.path.join(args.data_dir, 'val_set.json'),
        'test': os.path.join(args.data_dir, 'test_set.json'),
        'train_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
        'valid_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
        'test_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
    }

    img_path = {m: args.img_dir for m in ['train', 'valid', 'test']} if args.img_dir else None
    dep_path = {m: args.dep_dir for m in ['train', 'valid', 'test']} if args.dep_dir else None
    cap_path = {m: args.cap_path for m in ['train', 'valid', 'test']} if args.cap_path else None
    re_path = args.re_path or os.path.join(args.data_dir, 'rel2id.json')

    logger.info(args)

    # 数据加载
    processor = UMKEProcessor(data_path, re_path, args.bert_name, args.vit_name)
    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer

    test_dataset = UMKEDataset(processor, img_path, dep_path, cap_path, args, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    # 需要一些虚拟参数给 argparse
    args.num_epochs = 0
    args.lr = 0
    args.warmup_ratio = 0
    args.eval_begin_epoch = 0
    args.save_path = None
    args.notes = ""

    model = RelevanceAwareUnimoREModel(num_labels, tokenizer, args)
    state_dict = torch.load(args.load_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)

    # 鲁棒性评测
    if args.do_robust_eval:
        logger.info("\n" + "=" * 60)
        logger.info("ROBUSTNESS EVALUATION")
        logger.info("=" * 60)
        robust_results = robust_evaluate(
            model, test_dataloader, re_dict, args,
            perturbation_names=args.perturbations
        )

    # 校准评测
    if args.do_calibrate:
        logger.info("\n" + "=" * 60)
        logger.info("CALIBRATION EVALUATION")
        logger.info("=" * 60)

        # 在验证集上拟合温度
        valid_dataset = UMKEDataset(processor, img_path, dep_path, cap_path, args, mode='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        temp_scaling = fit_temperature(model, valid_dataloader, args)

        # 在测试集上评测
        calib_results = calibration_evaluate(model, test_dataloader, args, temperature=temp_scaling)

    # 如果都没开，跑 clean 评测
    if not args.do_robust_eval and not args.do_calibrate:
        logger.info("Running clean evaluation...")
        from eval.robust_eval import _evaluate_single
        result = _evaluate_single(model, test_dataloader, re_dict, args)
        logger.info("Clean F1: {:.4f}, Acc: {:.4f}, ECE: {:.4f}".format(
            result['micro_f1'], result['acc'], result['ece']))


if __name__ == "__main__":
    main()
