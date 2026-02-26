import torch
import random
import numpy as np


def shuffle_images(batch, seed=None):
    """图文错配: batch 内随机置换图像

    Args:
        batch: dict, 包含 'ent_imgs', 'org_image' 等字段
    Returns:
        batch: 修改后的 batch
    """
    bsz = batch['ent_imgs'].shape[0]
    if seed is not None:
        rng = np.random.RandomState(seed)
        perm = torch.tensor(rng.permutation(bsz))
    else:
        perm = torch.randperm(bsz)
    batch['ent_imgs'] = batch['ent_imgs'][perm]
    if 'org_image' in batch:
        batch['org_image'] = batch['org_image'][perm]
    if 'position' in batch:
        batch['position'] = batch['position'][perm]
    if 'ent_idx_head' in batch:
        batch['ent_idx_head'] = batch['ent_idx_head'][perm]
    if 'ent_idx_tail' in batch:
        batch['ent_idx_tail'] = batch['ent_idx_tail'][perm]
    return batch


def drop_boxes(batch, drop_prob=0.3):
    """对象框丢弃: 以概率 p 将对象图像置零

    Args:
        batch: dict
        drop_prob: 丢弃概率
    Returns:
        batch: 修改后的 batch
    """
    ent_imgs = batch['ent_imgs'].clone()  # [bsz, 12, C, H, W]
    bsz, num_obj = ent_imgs.shape[0], ent_imgs.shape[1]
    mask = torch.rand(bsz, num_obj) < drop_prob
    ent_imgs[mask] = 0.0
    batch['ent_imgs'] = ent_imgs
    return batch


def jitter_boxes(batch, sigma=0.05):
    """对象框抖动: 对 position 加高斯噪声

    Args:
        batch: dict, 需包含 'position'
        sigma: 噪声标准差
    Returns:
        batch: 修改后的 batch
    """
    if 'position' not in batch:
        return batch
    position = batch['position'].clone().float()
    noise = torch.randn_like(position) * sigma
    position = position + noise
    position = position.clamp(0.0, 1.0)
    batch['position'] = position
    return batch


def add_distractors(batch, num_distractors=2):
    """干扰框注入: 在 padding 位置填入随机噪声图像

    Args:
        batch: dict
        num_distractors: 注入的干扰框数量
    Returns:
        batch: 修改后的 batch
    """
    ent_imgs = batch['ent_imgs'].clone()  # [bsz, 12, C, H, W]
    bsz, num_obj, C, H, W = ent_imgs.shape

    for i in range(bsz):
        # 找到 padding 位置（全零的 slot）
        is_padding = (ent_imgs[i].view(num_obj, -1).abs().sum(dim=-1) == 0)
        padding_indices = is_padding.nonzero(as_tuple=True)[0]

        n_fill = min(num_distractors, len(padding_indices))
        for j in range(n_fill):
            idx = padding_indices[j].item()
            ent_imgs[i, idx] = torch.randn(C, H, W) * 0.5

    batch['ent_imgs'] = ent_imgs
    return batch


def caption_replace(batch, replace_ratio=0.3, vocab_size=30522, pad_token_id=0, special_token_ids=None):
    """Caption 替换: 随机替换 30% caption token

    替换 token_type_ids == 1 的区域（即 caption/text_pair 段）中的 token。

    Args:
        batch: dict
        replace_ratio: 替换比例
        vocab_size: 词表大小
        pad_token_id: PAD token id
        special_token_ids: 不替换的特殊 token id 集合
    Returns:
        batch: 修改后的 batch
    """
    if special_token_ids is None:
        special_token_ids = set()

    input_ids = batch['input_ids'].clone()
    token_type_ids = batch['token_type_ids']
    bsz, seq_len = input_ids.shape

    for i in range(bsz):
        cap_mask = (token_type_ids[i] == 1) & (input_ids[i] != pad_token_id)
        cap_indices = cap_mask.nonzero(as_tuple=True)[0]
        if len(cap_indices) == 0:
            continue
        n_replace = max(1, int(len(cap_indices) * replace_ratio))
        replace_indices = cap_indices[torch.randperm(len(cap_indices))[:n_replace]]
        for idx in replace_indices:
            if input_ids[i, idx].item() not in special_token_ids:
                input_ids[i, idx] = random.randint(1, vocab_size - 1)

    batch['input_ids'] = input_ids
    return batch


def caption_delete(batch, pad_token_id=0):
    """Caption 删除: 将 caption 段全部置为 [PAD]

    Args:
        batch: dict
        pad_token_id: PAD token id
    Returns:
        batch: 修改后的 batch
    """
    input_ids = batch['input_ids'].clone()
    token_type_ids = batch['token_type_ids']
    attention_mask = batch['attention_mask'].clone()
    bsz = input_ids.shape[0]

    for i in range(bsz):
        cap_mask = token_type_ids[i] == 1
        input_ids[i][cap_mask] = pad_token_id
        attention_mask[i][cap_mask] = 0

    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_mask
    return batch


# 扰动注册表
PERTURBATION_REGISTRY = {
    'shuffle_images': shuffle_images,
    'drop_boxes_0.3': lambda b: drop_boxes(b, drop_prob=0.3),
    'drop_boxes_0.5': lambda b: drop_boxes(b, drop_prob=0.5),
    'jitter_boxes': lambda b: jitter_boxes(b, sigma=0.05),
    'add_distractors': lambda b: add_distractors(b, num_distractors=2),
    'caption_replace': lambda b: caption_replace(b, replace_ratio=0.3),
    'caption_delete': caption_delete,
}
