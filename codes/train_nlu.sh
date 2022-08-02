
DATA_DIR="/home/heh/github/NeMo/codes/manifests_nlu_bio2"

EXP_NAME="nlu_intent_slot_classification"

# EXP_NAME="nlu_ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4_dec3_d2048h8_shareToken"
# dataset_manifest="${DATA_DIR}/test_nlu.json" \

CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/${EXP_NAME}/checkpoints"

# python checkpoint_averaging.py ${CKPT_DIR}

# NEMO_MODEL="${CKPT_DIR}/Conformer-Transformer-SLU2ASR-averaged.nemo"
NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"

CUDA_VISIBLE_DEVICES=0 python intent_slot_classification.py \
    --config-path="./configs" --config-name=intent_slot_classification_config \
    model.data_dir=$DATA_DIR \
    model.pretrained=$NEMO_MODEL \
    test_only=true \
    exp_manager.create_wandb_logger=false
