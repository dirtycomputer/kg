import os
import argparse
import logging
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.unimo_model import RelevanceAwareUnimoREModel
from datasets.umke_dataset import UMKEProcessor, UMKEDataset
from modules.trainer import Trainer
import warnings
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Relevance-Aware Unified Multimodal RE Training")

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

    # 训练
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--notes', default="", type=str)

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

    args = parser.parse_args()

    # VIB 总开关
    if not args.use_vib:
        args.use_seq_vib = False
        args.use_entity_vib = False

    set_seed(args.seed)

    # 数据路径构建
    data_path = {
        'train': os.path.join(args.data_dir, 'train_set.json'),
        'valid': os.path.join(args.data_dir, 'val_set.json'),
        'test': os.path.join(args.data_dir, 'test_set.json'),
        'train_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
        'valid_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
        'test_ent_dict': os.path.join(args.data_dir, 'pos_umke.json'),
    }

    img_path = None
    if args.img_dir:
        img_path = {m: args.img_dir for m in ['train', 'valid', 'test']}

    dep_path = None
    if args.dep_dir:
        dep_path = {m: args.dep_dir for m in ['train', 'valid', 'test']}

    cap_path = None
    if args.cap_path:
        cap_path = {m: args.cap_path for m in ['train', 'valid', 'test']}

    re_path = args.re_path or os.path.join(args.data_dir, 'rel2id.json')

    # 保存路径
    if args.save_path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_path = os.path.join(
            args.save_path,
            "{}_{}_{}_{}_{}" .format(args.dataset_name, args.batch_size, args.lr, args.notes, current_time)
        )
        os.makedirs(args.save_path, exist_ok=True)

    logger.info(args)

    # 数据加载
    processor = UMKEProcessor(data_path, re_path, args.bert_name, args.vit_name)

    train_dataset = UMKEDataset(processor, img_path, dep_path, cap_path, args, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valid_dataset = UMKEDataset(processor, img_path, dep_path, cap_path, args, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = UMKEDataset(processor, img_path, dep_path, cap_path, args, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer

    # 模型
    model = RelevanceAwareUnimoREModel(num_labels, tokenizer, args)
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    # 训练
    trainer = Trainer(
        train_data=train_dataloader, dev_data=valid_dataloader, test_data=test_dataloader,
        re_dict=re_dict, model=model, args=args, logger=logger, writer=None
    )
    trainer.train()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
