DATA_DIR="/home/heh/datasets/slurp_draco"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/2022-06-29_10-41-00/checkpoints"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/checkpoints"

# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_2LR_ep200"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d512h4"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_adapter"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4_dec3_d2048h8_freezeEnc"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_r2"
# EXP_NAME="ssl_en_conformer_xlarge_transformer_CosineAnneal_lr3e-4x2e-4_dec3_d2048h8"
# EXP_NAME="ssl_en_conformer_xlarge_transformer_CosineAnneal_lr3e-4x2e-4_dec3_d2048h8"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_asr3"
# EXP_NAME="stt_en_conformer_ctc_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_adapter"
EXP_NAME="stt_en_conformer_transducer_large_CosineAnneal_lr1e-3x5e-4_rnn3_ms10_asr3"

CKPT_DIR="/home/heh/github/NeMo-heh/codes/nemo_experiments/${EXP_NAME}/checkpoints"

# python checkpoint_averaging.py ${CKPT_DIR}

# NEMO_MODEL="${CKPT_DIR}/Conformer-Transformer-SLU2ASR-averaged.nemo"
NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"
CUDA_VISIBLE_DEVICES=1 python run_slurp_rnnt_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu2asr.json" \
    model_path=${NEMO_MODEL} \
    batch_size=32 \
    num_workers=8 \
    searcher.type="greedy" \
    searcher.beam_size=32

# dataset_manifest="${DATA_DIR}/test_slu2asr.json"
# dataset_manifest="evaluation_transcripts.json"
# dataset_manifest="${CKPT_DIR}/predictions.json"
