"""
综合测试脚本 — 覆盖所有新增模块的单元测试 + 冒烟测试

测试项:
1. RelevanceGate: 输出形状、r ∈ [0,1]、正样本 r > 负样本 r
2. GateContrastiveLoss: loss > 0、梯度正常
3. SequenceVIB: 输出形状、train/eval 行为差异、KL >= 0
4. EntityVIB + BetaNet: 自适应 β 方向正确（r低+u高 → β大）
5. CalibrationMetrics: ECE/Brier/NLL 用已知分布验证
6. TemperatureScaling: 拟合后 NLL 下降
7. noise_injection: 7 种扰动函数正确性
8. metrics: eval_result + compute_calibration_metrics
9. 端到端冒烟测试: 模拟 forward + backward（不需要预训练模型）
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, fn):
    global PASSED, FAILED, ERRORS
    try:
        fn()
        PASSED += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, traceback.format_exc()))
        print(f"  [FAIL] {name}: {e}")


# ============================================================
# Test 1: RelevanceGate
# ============================================================
print("\n=== Test Group 1: RelevanceGate ===")

from models.relevance_gate import RelevanceGate, GateContrastiveLoss


def test_gate_output_shape():
    gate = RelevanceGate(hidden_size=768)
    t_hidden = torch.randn(4, 128, 768)
    v_hidden = torch.randn(4, 12, 768)
    r, entropy_loss = gate(t_hidden, v_hidden)
    assert r.shape == (4, 1), f"Expected (4,1), got {r.shape}"
    assert entropy_loss.dim() == 0, "entropy_loss should be scalar"


def test_gate_r_range():
    gate = RelevanceGate(hidden_size=768)
    t_hidden = torch.randn(8, 128, 768)
    v_hidden = torch.randn(8, 12, 768)
    r, _ = gate(t_hidden, v_hidden)
    assert (r >= 0).all() and (r <= 1).all(), f"r out of [0,1]: min={r.min()}, max={r.max()}"


def test_gate_positive_vs_negative():
    """正样本 r 应该 > 负样本 r（训练几步后）"""
    torch.manual_seed(42)
    gate = RelevanceGate(hidden_size=128)
    contra = GateContrastiveLoss(neg_num=1)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)

    # 构造有区分度的正负样本
    t_hidden = torch.randn(16, 32, 128)
    v_hidden = t_hidden[:, :12, :] + torch.randn(16, 12, 128) * 0.1  # 正样本：相似
    v_neg = v_hidden.roll(shifts=1, dims=0)  # 负样本：不匹配

    for _ in range(50):
        loss = contra(gate, t_hidden, v_hidden)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    gate.eval()
    with torch.no_grad():
        r_pos, _ = gate(t_hidden, v_hidden)
        r_neg, _ = gate(t_hidden, v_neg)
    assert r_pos.mean() > r_neg.mean(), \
        f"Expected r_pos.mean ({r_pos.mean():.4f}) > r_neg.mean ({r_neg.mean():.4f})"


def test_gate_contrastive_loss():
    gate = RelevanceGate(hidden_size=768)
    contra = GateContrastiveLoss(neg_num=1)
    t_hidden = torch.randn(4, 128, 768)
    v_hidden = torch.randn(4, 12, 768)
    loss = contra(gate, t_hidden, v_hidden)
    assert loss.item() > 0, "Contrastive loss should be > 0"
    loss.backward()
    grad_count = sum(1 for p in gate.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0, "No gradients flowing through gate"


run_test("gate_output_shape", test_gate_output_shape)
run_test("gate_r_range", test_gate_r_range)
run_test("gate_positive_vs_negative", test_gate_positive_vs_negative)
run_test("gate_contrastive_loss", test_gate_contrastive_loss)


# ============================================================
# Test 2: SequenceVIB
# ============================================================
print("\n=== Test Group 2: SequenceVIB ===")

from models.adaptive_vib import SequenceVIB, EntityVIB, BetaNet


def test_seq_vib_shape():
    vib = SequenceVIB(hidden_size=768)
    hidden = torch.randn(4, 128, 768)
    z, kl = vib(hidden, mode='train')
    assert z.shape == (4, 128, 768), f"Expected (4,128,768), got {z.shape}"
    assert kl.dim() == 0, "KL should be scalar"
    assert kl.item() >= 0, f"KL should be >= 0, got {kl.item()}"


def test_seq_vib_eval_deterministic():
    vib = SequenceVIB(hidden_size=768)
    vib.eval()
    hidden = torch.randn(4, 128, 768)
    z1, _ = vib(hidden, mode='eval')
    z2, _ = vib(hidden, mode='eval')
    assert torch.allclose(z1, z2), "Eval mode should be deterministic"


def test_seq_vib_train_stochastic():
    vib = SequenceVIB(hidden_size=768)
    vib.train()
    hidden = torch.randn(4, 128, 768)
    torch.manual_seed(1)
    z1, _ = vib(hidden, mode='train')
    torch.manual_seed(2)
    z2, _ = vib(hidden, mode='train')
    assert not torch.allclose(z1, z2), "Train mode should be stochastic"


run_test("seq_vib_shape", test_seq_vib_shape)
run_test("seq_vib_eval_deterministic", test_seq_vib_eval_deterministic)
run_test("seq_vib_train_stochastic", test_seq_vib_train_stochastic)


# ============================================================
# Test 3: EntityVIB + BetaNet
# ============================================================
print("\n=== Test Group 3: EntityVIB + BetaNet ===")


def test_entity_vib_shape():
    evib = EntityVIB(hidden_size=768, num_classes=10)
    entity_hidden = torch.randn(4, 768)
    r = torch.rand(4, 1)
    logits = torch.randn(4, 10)
    z, wkl, beta = evib(entity_hidden, r, mode='train', preliminary_logits=logits)
    assert z.shape == (4, 768), f"Expected (4,768), got {z.shape}"
    assert wkl.dim() == 0, "weighted_kl should be scalar"
    assert beta.shape == (4, 1), f"Expected (4,1), got {beta.shape}"
    assert (beta >= 1e-4).all() and (beta <= 1.0).all(), f"beta out of range: {beta}"


def test_beta_net_direction():
    """r 低 + u 高 → β 大; r 高 + u 低 → β 小"""
    torch.manual_seed(42)
    beta_net = BetaNet()

    # 训练 beta_net 使其学到正确方向
    optimizer = torch.optim.Adam(beta_net.parameters(), lr=1e-2)
    for _ in range(200):
        r_low = torch.rand(32, 1) * 0.3       # r 低
        u_high = 0.7 + torch.rand(32, 1) * 0.3  # u 高
        r_high = 0.7 + torch.rand(32, 1) * 0.3  # r 高
        u_low = torch.rand(32, 1) * 0.3         # u 低

        beta_high_target = beta_net(r_low, u_high)
        beta_low_target = beta_net(r_high, u_low)

        # 希望 beta_high > beta_low
        loss = F.relu(beta_low_target - beta_high_target + 0.1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    beta_net.eval()
    with torch.no_grad():
        beta_noisy = beta_net(torch.tensor([[0.1]]), torch.tensor([[0.9]]))
        beta_clean = beta_net(torch.tensor([[0.9]]), torch.tensor([[0.1]]))
    assert beta_noisy.item() > beta_clean.item(), \
        f"Expected beta_noisy ({beta_noisy.item():.4f}) > beta_clean ({beta_clean.item():.4f})"


def test_entity_vib_gradient():
    evib = EntityVIB(hidden_size=768, num_classes=10)
    entity_hidden = torch.randn(4, 768, requires_grad=True)
    r = torch.rand(4, 1)
    logits = torch.randn(4, 10)
    z, wkl, beta = evib(entity_hidden, r, mode='train', preliminary_logits=logits)
    loss = z.sum() + wkl
    loss.backward()
    assert entity_hidden.grad is not None, "No gradient on entity_hidden"
    assert entity_hidden.grad.abs().sum() > 0, "Zero gradient on entity_hidden"


run_test("entity_vib_shape", test_entity_vib_shape)
run_test("beta_net_direction", test_beta_net_direction)
run_test("entity_vib_gradient", test_entity_vib_gradient)


# ============================================================
# Test 4: CalibrationMetrics + TemperatureScaling
# ============================================================
print("\n=== Test Group 4: Calibration ===")

from models.calibration import TemperatureScaling, CalibrationMetrics


def test_ece_perfect_calibration():
    """完美校准的模型 ECE 应该接近 0"""
    torch.manual_seed(42)
    num_classes = 5
    N = 1000
    labels = torch.randint(0, num_classes, (N,))
    # 构造完美校准的概率：one-hot
    probs = F.one_hot(labels, num_classes).float() * 0.95 + 0.01
    ece = CalibrationMetrics.ece(probs, labels)
    assert ece < 0.1, f"Perfect calibration ECE should be < 0.1, got {ece:.4f}"


def test_brier_score_range():
    probs = torch.softmax(torch.randn(100, 5), dim=1)
    labels = torch.randint(0, 5, (100,))
    brier = CalibrationMetrics.brier_score(probs, labels)
    assert 0 <= brier <= 2.0, f"Brier score out of range: {brier}"


def test_nll_positive():
    probs = torch.softmax(torch.randn(100, 5), dim=1)
    labels = torch.randint(0, 5, (100,))
    nll = CalibrationMetrics.nll(probs, labels)
    assert nll > 0, f"NLL should be > 0, got {nll}"


def test_compute_all():
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    result = CalibrationMetrics.compute_all(logits, labels)
    assert 'ece' in result and 'brier' in result and 'nll' in result
    assert all(isinstance(v, float) for v in result.values())


def test_temperature_scaling_fit():
    """温度缩放后 NLL 应该下降"""
    torch.manual_seed(42)
    logits = torch.randn(200, 5) * 3  # 过度自信的 logits
    labels = torch.randint(0, 5, (200,))

    nll_before = F.cross_entropy(logits, labels).item()

    ts = TemperatureScaling()
    optimal_t = ts.fit(logits, labels)

    scaled_logits = ts(logits)
    nll_after = F.cross_entropy(scaled_logits, labels).item()

    assert nll_after <= nll_before + 1e-4, \
        f"NLL should decrease after calibration: before={nll_before:.4f}, after={nll_after:.4f}"
    assert optimal_t > 0, f"Temperature should be > 0, got {optimal_t}"


run_test("ece_perfect_calibration", test_ece_perfect_calibration)
run_test("brier_score_range", test_brier_score_range)
run_test("nll_positive", test_nll_positive)
run_test("compute_all", test_compute_all)
run_test("temperature_scaling_fit", test_temperature_scaling_fit)


# ============================================================
# Test 5: noise_injection
# ============================================================
print("\n=== Test Group 5: Noise Injection ===")

from datasets.noise_injection import (
    shuffle_images, drop_boxes, jitter_boxes,
    add_distractors, caption_replace, caption_delete,
    PERTURBATION_REGISTRY
)


def make_fake_batch(bsz=4, seq_len=64, num_obj=12, C=3, H=64, W=64):
    return {
        'input_ids': torch.randint(1, 30000, (bsz, seq_len)),
        'token_type_ids': torch.cat([
            torch.zeros(bsz, seq_len // 2, dtype=torch.long),
            torch.ones(bsz, seq_len // 2, dtype=torch.long)
        ], dim=1),
        'attention_mask': torch.ones(bsz, seq_len, dtype=torch.long),
        'ent_imgs': torch.randn(bsz, num_obj, C, H, W),
        'org_image': torch.randn(bsz, C, 224, 224),
        'position': torch.rand(bsz, num_obj, 5),
        'ent_idx_head': torch.randint(0, num_obj, (bsz,)),
        'ent_idx_tail': torch.randint(0, num_obj, (bsz,)),
    }


def test_shuffle_images():
    batch = make_fake_batch()
    orig_imgs = batch['ent_imgs'].clone()
    batch = shuffle_images(batch, seed=42)
    assert batch['ent_imgs'].shape == orig_imgs.shape
    # 至少有一个样本被移动了
    assert not torch.allclose(batch['ent_imgs'], orig_imgs), "Shuffle should change order"


def test_drop_boxes():
    batch = make_fake_batch()
    batch = drop_boxes(batch, drop_prob=0.5)
    # 应该有一些 slot 被置零
    zero_count = (batch['ent_imgs'].view(4, 12, -1).abs().sum(-1) == 0).sum()
    assert zero_count > 0, "drop_boxes should zero out some slots"


def test_jitter_boxes():
    batch = make_fake_batch()
    orig_pos = batch['position'].clone()
    batch = jitter_boxes(batch, sigma=0.05)
    assert not torch.allclose(batch['position'], orig_pos), "Jitter should change positions"
    assert (batch['position'] >= 0).all() and (batch['position'] <= 1).all(), "Position out of [0,1]"


def test_add_distractors():
    batch = make_fake_batch()
    # 手动将最后 4 个 slot 置零（模拟 padding）
    batch['ent_imgs'][:, -4:] = 0
    batch = add_distractors(batch, num_distractors=2)
    # 检查至少有一些 padding 被填充了
    filled = (batch['ent_imgs'][:, -4:].view(4, 4, -1).abs().sum(-1) > 0).sum()
    assert filled > 0, "add_distractors should fill some padding slots"


def test_caption_replace():
    batch = make_fake_batch()
    orig_ids = batch['input_ids'].clone()
    batch = caption_replace(batch, replace_ratio=0.3)
    # token_type_ids == 1 的区域应该有变化
    cap_mask = batch['token_type_ids'] == 1
    changed = (batch['input_ids'] != orig_ids) & cap_mask
    assert changed.sum() > 0, "caption_replace should change some caption tokens"


def test_caption_delete():
    batch = make_fake_batch()
    batch = caption_delete(batch, pad_token_id=0)
    cap_mask = batch['token_type_ids'] == 1
    assert (batch['input_ids'][cap_mask] == 0).all(), "caption_delete should zero all caption tokens"
    assert (batch['attention_mask'][cap_mask] == 0).all(), "caption_delete should zero attention for captions"


def test_perturbation_registry():
    assert len(PERTURBATION_REGISTRY) == 7, f"Expected 7 perturbations, got {len(PERTURBATION_REGISTRY)}"
    batch = make_fake_batch()
    batch['ent_imgs'][:, -4:] = 0  # padding for add_distractors
    for name, fn in PERTURBATION_REGISTRY.items():
        b = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        result = fn(b)
        assert isinstance(result, dict), f"{name} should return dict"
        assert 'ent_imgs' in result or 'input_ids' in result, f"{name} should modify batch"


run_test("shuffle_images", test_shuffle_images)
run_test("drop_boxes", test_drop_boxes)
run_test("jitter_boxes", test_jitter_boxes)
run_test("add_distractors", test_add_distractors)
run_test("caption_replace", test_caption_replace)
run_test("caption_delete", test_caption_delete)
run_test("perturbation_registry", test_perturbation_registry)


# ============================================================
# Test 6: metrics
# ============================================================
print("\n=== Test Group 6: Metrics ===")

import logging
from modules.metrics import eval_result, compute_calibration_metrics

test_logger = logging.getLogger("test")
test_logger.setLevel(logging.WARNING)


def test_eval_result_perfect():
    re_dict = {'none': 0, 'rel_A': 1, 'rel_B': 2}
    true_labels = [1, 1, 2, 2, 0]
    pred_labels = [1, 1, 2, 2, 0]
    result = eval_result(true_labels, pred_labels, re_dict, test_logger)
    assert result['acc'] == 1.0, f"Perfect prediction acc should be 1.0, got {result['acc']}"
    assert result['micro_f1'] == 1.0, f"Perfect prediction F1 should be 1.0, got {result['micro_f1']}"


def test_eval_result_partial():
    re_dict = {'none': 0, 'rel_A': 1, 'rel_B': 2}
    true_labels = [1, 1, 2, 0]
    pred_labels = [1, 2, 2, 0]
    result = eval_result(true_labels, pred_labels, re_dict, test_logger)
    assert 0 < result['micro_f1'] < 1.0, f"Partial prediction F1 should be in (0,1), got {result['micro_f1']}"


def test_compute_calibration_metrics():
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    result = compute_calibration_metrics(logits, labels)
    assert 'ece' in result and 'brier' in result and 'nll' in result
    assert result['ece'] >= 0
    assert result['brier'] >= 0
    assert result['nll'] > 0


run_test("eval_result_perfect", test_eval_result_perfect)
run_test("eval_result_partial", test_eval_result_partial)
run_test("compute_calibration_metrics", test_compute_calibration_metrics)


# ============================================================
# Test 7: 端到端冒烟测试（模拟 forward + backward，不需要预训练模型）
# ============================================================
print("\n=== Test Group 7: End-to-End Smoke Test ===")


def test_e2e_forward_backward():
    """
    模拟完整的 forward + backward 流程。
    不加载预训练模型，直接构造各子模块并组装。
    """
    bsz = 2
    seq_len = 32
    hidden_size = 128
    num_labels = 5
    num_obj = 12

    # 模拟 encoder 输出
    t_output = torch.randn(bsz, seq_len, hidden_size, requires_grad=True)
    v_output = torch.randn(bsz, num_obj, hidden_size, requires_grad=True)

    # 模块
    seq_vib_text = SequenceVIB(hidden_size)
    seq_vib_vision = SequenceVIB(hidden_size)
    gate = RelevanceGate(hidden_size)
    gate_contra = GateContrastiveLoss(neg_num=1)
    entity_vib_head = EntityVIB(hidden_size, num_classes=num_labels)
    entity_vib_tail = EntityVIB(hidden_size, num_classes=num_labels)
    classifier = nn.Linear(hidden_size * 2, num_labels)
    labels = torch.randint(0, num_labels, (bsz,))

    # Forward
    t_out, t_kl = seq_vib_text(t_output, mode='train')
    v_out, v_kl = seq_vib_vision(v_output, mode='train')

    r, entropy_loss = gate(t_out, v_out)
    gated_v = r.unsqueeze(-1) * v_out
    gate_loss = gate_contra(gate, t_out, v_out)

    # 模拟实体提取（取第一个 token 和最后一个 token）
    head_hidden = t_out[:, 0, :]
    tail_hidden = t_out[:, -1, :]
    entity_state = torch.cat([head_hidden, tail_hidden], dim=-1)

    # 初步分类
    with torch.no_grad():
        prelim_logits = classifier(entity_state)

    z_head, kl_head, beta_h = entity_vib_head(head_hidden, r, 'train', prelim_logits)
    z_tail, kl_tail, beta_t = entity_vib_tail(tail_hidden, r, 'train', prelim_logits)
    final_entity = torch.cat([z_head, z_tail], dim=-1)

    logits = classifier(final_entity)

    # Loss
    L_ce = F.cross_entropy(logits, labels)
    beta_seq = 0.01
    L_seq_vib = beta_seq * (t_kl + v_kl)
    L_entity_vib = kl_head + kl_tail
    L_gate = 0.5 * gate_loss + 0.1 * entropy_loss
    L_total = L_ce + L_seq_vib + L_entity_vib + L_gate

    assert L_total.dim() == 0, "Total loss should be scalar"
    assert not torch.isnan(L_total), "Total loss is NaN"
    assert not torch.isinf(L_total), "Total loss is Inf"

    # Backward
    L_total.backward()

    assert t_output.grad is not None, "No gradient on t_output"
    assert v_output.grad is not None, "No gradient on v_output"
    assert t_output.grad.abs().sum() > 0, "Zero gradient on t_output"
    assert v_output.grad.abs().sum() > 0, "Zero gradient on v_output"

    # 检查所有模块都有梯度
    for name, module in [('seq_vib_text', seq_vib_text), ('seq_vib_vision', seq_vib_vision),
                          ('gate', gate), ('entity_vib_head', entity_vib_head),
                          ('entity_vib_tail', entity_vib_tail), ('classifier', classifier)]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert has_grad, f"No gradient flowing through {name}"


def test_e2e_eval_mode():
    """Eval 模式下应该是确定性的"""
    bsz = 2
    hidden_size = 128
    num_labels = 5

    seq_vib = SequenceVIB(hidden_size)
    entity_vib = EntityVIB(hidden_size, num_classes=num_labels)

    hidden = torch.randn(bsz, 32, hidden_size)
    entity = torch.randn(bsz, hidden_size)
    r = torch.rand(bsz, 1)
    logits = torch.randn(bsz, num_labels)

    z1, _ = seq_vib(hidden, mode='eval')
    z2, _ = seq_vib(hidden, mode='eval')
    assert torch.allclose(z1, z2), "Eval mode SequenceVIB should be deterministic"

    ez1, _, _ = entity_vib(entity, r, mode='eval', preliminary_logits=logits)
    ez2, _, _ = entity_vib(entity, r, mode='eval', preliminary_logits=logits)
    assert torch.allclose(ez1, ez2), "Eval mode EntityVIB should be deterministic"


def test_e2e_loss_components():
    """验证各 loss 分项都是有限正数"""
    bsz = 4
    hidden_size = 128
    num_labels = 5

    t_out = torch.randn(bsz, 32, hidden_size)
    v_out = torch.randn(bsz, 12, hidden_size)

    seq_vib = SequenceVIB(hidden_size)
    gate = RelevanceGate(hidden_size)
    contra = GateContrastiveLoss()

    _, t_kl = seq_vib(t_out, 'train')
    _, v_kl = seq_vib(v_out, 'train')
    r, ent_loss = gate(t_out, v_out)
    g_loss = contra(gate, t_out, v_out)

    for name, val in [('t_kl', t_kl), ('v_kl', v_kl), ('entropy_loss', ent_loss), ('gate_loss', g_loss)]:
        assert torch.isfinite(val), f"{name} is not finite: {val}"


run_test("e2e_forward_backward", test_e2e_forward_backward)
run_test("e2e_eval_mode", test_e2e_eval_mode)
run_test("e2e_loss_components", test_e2e_loss_components)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
print("=" * 60)

if ERRORS:
    print("\nFailed tests:")
    for name, tb in ERRORS:
        print(f"\n--- {name} ---")
        print(tb)

sys.exit(0 if FAILED == 0 else 1)
