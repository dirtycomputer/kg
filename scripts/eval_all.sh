#!/bin/bash
# 一键评测脚本 - Clean + Robust + Calibration

cd "$(dirname "$0")/../src" || exit

# === 配置区 ===
DATASET="UMKE"
DATA_DIR="/data/umke_partner"
IMG_DIR="/data/UMKE_IMG"
DEP_DIR="/data/depth_data_umke"
CAP_PATH="/data/umke_partner/cap_qwenvl_simply.json"
RE_PATH="/data/lxk2019/UMREF/MORE_test/rel2id_umke_partner.json"

BERT_NAME="bert-base-uncased"
VIT_NAME="openai/clip-vit-base-patch32"

BATCH_SIZE=16
MAX_SEQ=128
SEED=1

# 模型 checkpoint 路径（需修改为实际路径）
LOAD_PATH="../checkpoints/best_model.pth"

echo "=========================================="
echo "  Step 1: Robustness Evaluation"
echo "=========================================="
python run_eval.py \
    --dataset_name ${DATASET} \
    --data_dir ${DATA_DIR} \
    --img_dir ${IMG_DIR} \
    --dep_dir ${DEP_DIR} \
    --cap_path ${CAP_PATH} \
    --re_path ${RE_PATH} \
    --bert_name ${BERT_NAME} \
    --vit_name ${VIT_NAME} \
    --batch_size ${BATCH_SIZE} \
    --max_seq ${MAX_SEQ} \
    --seed ${SEED} \
    --load_path ${LOAD_PATH} \
    --use_box \
    --use_cap \
    --use_dep \
    --use_gate \
    --use_vib \
    --use_seq_vib \
    --use_entity_vib \
    --do_robust_eval

echo ""
echo "=========================================="
echo "  Step 2: Calibration Evaluation"
echo "=========================================="
python run_calibrate.py \
    --dataset_name ${DATASET} \
    --data_dir ${DATA_DIR} \
    --img_dir ${IMG_DIR} \
    --dep_dir ${DEP_DIR} \
    --cap_path ${CAP_PATH} \
    --re_path ${RE_PATH} \
    --bert_name ${BERT_NAME} \
    --vit_name ${VIT_NAME} \
    --batch_size ${BATCH_SIZE} \
    --max_seq ${MAX_SEQ} \
    --seed ${SEED} \
    --load_path ${LOAD_PATH} \
    --use_box \
    --use_cap \
    --use_dep \
    --use_gate \
    --use_vib \
    --use_seq_vib \
    --use_entity_vib \
    --save_temp_path "../checkpoints/temperature.pt"

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="
