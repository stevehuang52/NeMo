DATA_DIR="/home/heh/datasets/slurp_draco"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/2022-06-29_10-41-00/checkpoints"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/checkpoints"

# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_2LR_ep200"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d512h4"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_adapter"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4_dec3_d2048h8_freezeEnc"
EXP_NAME="nlu_ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4_dec3_d2048h8"

CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/${EXP_NAME}/checkpoints"

# python checkpoint_averaging.py ${CKPT_DIR}

# NEMO_MODEL="${CKPT_DIR}/Conformer-Transformer-SLU2ASR-averaged.nemo"
NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"

CUDA_VISIBLE_DEVICES=1 python run_slurp_eval_nlu.py \
    dataset_manifest="${DATA_DIR}/test_nlu.json" \
    model_path=${NEMO_MODEL} \
    batch_size=32 \
    num_workers=8 \
    searcher.type="beam" \
    searcher.beam_size=32 \
    searcher.temperature=1.25 \
    only_score_manifest=false


# dataset_manifest="${DATA_DIR}/test_slu2asr.json"
# dataset_manifest="evaluation_transcripts.json"
