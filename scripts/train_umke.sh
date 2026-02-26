#!/bin/bash
# 一键训练脚本 - Relevance-Aware Unified Multimodal RE

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
LR=2e-5
EPOCHS=30
EVAL_BEGIN=16
SEED=1
MAX_SEQ=128

SAVE_PATH="../checkpoints"

# === 超参数 ===
BETA_SEQ=0.01
ALPHA_GATE=0.5
LAMBDA_ENTROPY=0.1

NOTES="gate_vib_full"

python run_train.py \
    --dataset_name ${DATASET} \
    --data_dir ${DATA_DIR} \
    --img_dir ${IMG_DIR} \
    --dep_dir ${DEP_DIR} \
    --cap_path ${CAP_PATH} \
    --re_path ${RE_PATH} \
    --bert_name ${BERT_NAME} \
    --vit_name ${VIT_NAME} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_epochs ${EPOCHS} \
    --eval_begin_epoch ${EVAL_BEGIN} \
    --seed ${SEED} \
    --max_seq ${MAX_SEQ} \
    --save_path ${SAVE_PATH} \
    --notes ${NOTES} \
    --use_box \
    --use_cap \
    --use_dep \
    --use_gate \
    --use_vib \
    --use_seq_vib \
    --use_entity_vib \
    --beta_seq ${BETA_SEQ} \
    --alpha_gate ${ALPHA_GATE} \
    --lambda_entropy ${LAMBDA_ENTROPY}
