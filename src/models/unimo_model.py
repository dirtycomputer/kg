import sys
sys.path.append("..")
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, CLIPConfig, BertModel
from .modeling_unimo import UnimoModel
from .modeling_clip import CLIPModel
from .relevance_gate import RelevanceGate, GateContrastiveLoss
from .adaptive_vib import SequenceVIB, EntityVIB
import numpy as np


class RelevanceAwareUnimoREModel(nn.Module):
    """相关性感知统一多模态关系抽取模型

    在 REMOTE 的 UnimoREModel 基础上集成:
    1. RelevanceGate: 相关性感知门控
    2. SequenceVIB: 序列级变分信息瓶颈
    3. EntityVIB: 实体级自适应变分信息瓶颈
    """

    def __init__(self, num_labels, tokenizer, args):
        super().__init__()
        self.args = args
        self.num_labels = num_labels

        # 加载预训练模型
        clip_model = CLIPModel.from_pretrained(args.vit_name, ignore_mismatched_sizes=True)
        clip_vit = clip_model.vision_model
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        bert_model_dict = bert.state_dict()

        self.vision_config = vision_config
        self.text_config = text_config
        hidden_size = text_config.hidden_size  # 768

        vision_config.device = args.device
        self.model = UnimoModel(vision_config, text_config)

        # 加载预训练权重
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        avg_conv_weight = torch.mean(clip_model_dict['embeddings.patch_embedding.weight'], dim=1).unsqueeze(1)
        clip_model_dict['embeddings.depth_embedding.weight'] = torch.tensor(avg_conv_weight.data.clone())

        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]

        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))

        # 分类器
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.mm_linear = nn.Linear(hidden_size * 2, hidden_size)

        # 实体标记 token ids
        self.head_entity_start = tokenizer.convert_tokens_to_ids("<s1>")
        self.tail_entity_start = tokenizer.convert_tokens_to_ids("<s2>")
        self.head_object_start = tokenizer.convert_tokens_to_ids("<o1>")
        self.tail_object_start = tokenizer.convert_tokens_to_ids("<o2>")
        self.head_knowledge_start = tokenizer.convert_tokens_to_ids("<k1>")
        self.tail_knowledge_start = tokenizer.convert_tokens_to_ids("<k2>")
        self.tokenizer = tokenizer

        # === 新增模块 ===
        # 相关性门控
        self.use_gate = getattr(args, 'use_gate', True)
        if self.use_gate:
            self.relevance_gate = RelevanceGate(hidden_size)
            self.gate_contrastive = GateContrastiveLoss(neg_num=1)

        # 序列级 VIB
        self.use_seq_vib = getattr(args, 'use_seq_vib', True)
        if self.use_seq_vib:
            self.seq_vib_text = SequenceVIB(hidden_size)
            self.seq_vib_vision = SequenceVIB(hidden_size)

        # 实体级 VIB
        self.use_entity_vib = getattr(args, 'use_entity_vib', True)
        if self.use_entity_vib:
            self.entity_vib_head = EntityVIB(hidden_size, num_classes=num_labels)
            self.entity_vib_tail = EntityVIB(hidden_size, num_classes=num_labels)

        # 超参数
        self.beta_seq = getattr(args, 'beta_seq', 0.01)
        self.alpha_gate = getattr(args, 'alpha_gate', 0.5)
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.1)

    def extract_entities(self, output, vision_output, input_ids, ent_idx_head, ent_idx_tail):
        """提取 head/tail 实体表示

        从文本输出和视觉输出中提取实体隐藏状态，
        根据实体类型（文本实体/视觉实体）选择对应的表示。

        Returns:
            entity_hidden_state: [bsz, hidden_size * 2]
        """
        bsz, seq_len, hidden_size = output.shape
        entity_hidden_state = torch.zeros(bsz, hidden_size * 2, device=output.device)

        for i in range(bsz):
            # 文本实体位置
            head_entity_idx = input_ids[i].eq(self.head_entity_start).nonzero()
            head_entity_idx = head_entity_idx.item() if head_entity_idx.numel() > 0 else -1

            tail_entity_idx = input_ids[i].eq(self.tail_entity_start).nonzero()
            tail_entity_idx = tail_entity_idx.item() if tail_entity_idx.numel() > 0 else -1

            # 文本实体隐藏状态
            head_entity_hidden = output[i, head_entity_idx, :].squeeze() if head_entity_idx != -1 else torch.zeros(hidden_size, device=output.device)
            tail_entity_hidden = output[i, tail_entity_idx, :].squeeze() if tail_entity_idx != -1 else torch.zeros(hidden_size, device=output.device)

            # 视觉实体隐藏状态
            vision_hidden_head = vision_output[i, ent_idx_head[i], :] if ent_idx_head[i] != -1 else torch.zeros(hidden_size, device=output.device)
            vision_hidden_tail = vision_output[i, ent_idx_tail[i], :] if ent_idx_tail[i] != -1 else torch.zeros(hidden_size, device=output.device)

            # 知识实体
            head_knowledge_idx = input_ids[i].eq(self.head_knowledge_start).nonzero()
            tail_knowledge_idx = input_ids[i].eq(self.tail_knowledge_start).nonzero()
            head_knowledge_idx = head_knowledge_idx.item() if head_knowledge_idx.numel() > 0 else -1
            tail_knowledge_idx = tail_knowledge_idx.item() if tail_knowledge_idx.numel() > 0 else -1

            head_knowledge_hidden = output[i, head_knowledge_idx, :].squeeze() if head_knowledge_idx != -1 else torch.zeros(hidden_size, device=output.device)
            tail_knowledge_hidden = output[i, tail_knowledge_idx, :].squeeze() if tail_knowledge_idx != -1 else torch.zeros(hidden_size, device=output.device)

            if self.args.use_cap:
                # Caption 模式下的多模态融合
                head_object_idx = input_ids[i].eq(self.head_object_start).nonzero()
                head_object_idx = head_object_idx.item() if head_object_idx.numel() > 0 else -1
                tail_object_idx = input_ids[i].eq(self.tail_object_start).nonzero()
                tail_object_idx = tail_object_idx.item() if tail_object_idx.numel() > 0 else -1

                head_object_hidden = output[i, head_object_idx, :].squeeze() if head_object_idx != -1 else torch.zeros(hidden_size, device=output.device)
                tail_object_hidden = output[i, tail_object_idx, :].squeeze() if tail_object_idx != -1 else torch.zeros(hidden_size, device=output.device)

                # Head 融合
                mm_head_hidden = []
                if head_entity_idx != -1 and ent_idx_head[i] == -1:
                    head_entity_hidden = self.mm_linear(torch.cat([head_entity_hidden, head_knowledge_hidden], dim=-1))
                elif head_entity_idx == -1 and ent_idx_head[i] != -1 and head_object_idx != -1:
                    mm_head_hidden = self.mm_linear(torch.cat([vision_hidden_head, head_object_hidden], dim=-1))

                # Tail 融合
                mm_tail_hidden = []
                if tail_entity_idx != -1 and ent_idx_tail[i] == -1:
                    tail_entity_hidden = self.mm_linear(torch.cat([tail_entity_hidden, tail_knowledge_hidden], dim=-1))
                elif tail_entity_idx == -1 and ent_idx_tail[i] != -1 and tail_object_idx != -1:
                    mm_tail_hidden = self.mm_linear(torch.cat([vision_hidden_tail, tail_object_hidden], dim=-1))

                if isinstance(mm_head_hidden, list):
                    mm_head_hidden = torch.tensor(mm_head_hidden, device=output.device)
                if isinstance(mm_tail_hidden, list):
                    mm_tail_hidden = torch.tensor(mm_tail_hidden, device=output.device)

                if mm_head_hidden.numel() > 0 and mm_tail_hidden.numel() > 0:
                    entity_hidden_state[i] = torch.cat([mm_head_hidden, mm_tail_hidden], dim=-1)
                elif mm_head_hidden.numel() == 0 and mm_tail_hidden.numel() > 0:
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, mm_tail_hidden], dim=-1)
                elif mm_head_hidden.numel() > 0 and mm_tail_hidden.numel() == 0:
                    entity_hidden_state[i] = torch.cat([mm_head_hidden, tail_entity_hidden], dim=-1)
                else:
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, tail_entity_hidden], dim=-1)
            else:
                # 非 Caption 模式
                if head_entity_idx != -1 and tail_entity_idx != -1:
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, tail_entity_hidden], dim=-1)
                elif head_entity_idx != -1 and ent_idx_tail[i] != -1:
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, vision_hidden_tail], dim=-1)
                elif ent_idx_head[i] != -1 and tail_entity_idx != -1:
                    entity_hidden_state[i] = torch.cat([vision_hidden_head, tail_entity_hidden], dim=-1)
                elif ent_idx_head[i] != -1 and ent_idx_tail[i] != -1:
                    entity_hidden_state[i] = torch.cat([vision_hidden_head, vision_hidden_tail], dim=-1)

        return entity_hidden_state

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            org_image=None,
            ent_imgs=None,
            ent_idx_head=None,
            ent_idx_tail=None,
            position=None,
            mode='train',
    ):
        # 1. UnimoEncoder
        t_output, v_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            aux_values=ent_imgs,
            position=position,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # t_output: [bsz, seq_len, 768], v_output: [bsz, 12, 768]

        # 2. 序列级 VIB
        t_seq_kl, v_seq_kl = torch.tensor(0.0, device=t_output.device), torch.tensor(0.0, device=t_output.device)
        if self.use_seq_vib:
            t_output, t_seq_kl = self.seq_vib_text(t_output, mode)
            v_output, v_seq_kl = self.seq_vib_vision(v_output, mode)

        # 3. 相关性门控
        gate_entropy_loss = torch.tensor(0.0, device=t_output.device)
        gate_contra_loss = torch.tensor(0.0, device=t_output.device)
        r = torch.ones(t_output.shape[0], 1, device=t_output.device)  # 默认全通

        if self.use_gate:
            r, gate_entropy_loss = self.relevance_gate(t_output, v_output)
            gated_v_output = r.unsqueeze(-1) * v_output
            # 4. 门控对比损失（仅训练时）
            if mode == 'train':
                gate_contra_loss = self.gate_contrastive(self.relevance_gate, t_output, v_output)
        else:
            gated_v_output = v_output

        # 5. 实体提取（用 gated_v_output）
        entity_hidden_state = self.extract_entities(
            t_output, gated_v_output, input_ids, ent_idx_head, ent_idx_tail)
        # entity_hidden_state: [bsz, hidden_size * 2]

        hidden_size = self.text_config.hidden_size
        head_hidden = entity_hidden_state[:, :hidden_size]
        tail_hidden = entity_hidden_state[:, hidden_size:]

        # 6. 实体级自适应 VIB
        kl_head = torch.tensor(0.0, device=t_output.device)
        kl_tail = torch.tensor(0.0, device=t_output.device)

        if self.use_entity_vib:
            # 初步分类（用于计算不确定性）
            with torch.no_grad():
                preliminary_logits = self.classifier(entity_hidden_state)

            # 7. 实体级 VIB
            z_head, kl_head, beta_h = self.entity_vib_head(head_hidden, r, mode, preliminary_logits)
            z_tail, kl_tail, beta_t = self.entity_vib_tail(tail_hidden, r, mode, preliminary_logits)
            final_entity = torch.cat([z_head, z_tail], dim=-1)
        else:
            final_entity = entity_hidden_state

        # 8. 最终分类
        logits = self.classifier(final_entity)

        if labels is not None:
            # 9. Loss 组合
            L_ce = F.cross_entropy(logits, labels.view(-1))
            L_seq_vib = self.beta_seq * (t_seq_kl + v_seq_kl)
            L_entity_vib = kl_head + kl_tail  # 已内含自适应 β
            L_gate = self.alpha_gate * gate_contra_loss + self.lambda_entropy * gate_entropy_loss
            L_total = L_ce + L_seq_vib + L_entity_vib + L_gate

            # 返回分项 loss 用于日志
            loss_dict = {
                'total': L_total,
                'ce': L_ce,
                'seq_vib': L_seq_vib,
                'entity_vib': L_entity_vib,
                'gate_contra': gate_contra_loss,
                'gate_entropy': gate_entropy_loss,
            }
            return L_total, logits, loss_dict

        return logits
