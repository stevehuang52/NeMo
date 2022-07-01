DATA_DIR="/home/heh/datasets/slurp_draco"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/2022-06-29_10-41-00/checkpoints"
CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/checkpoints"

# python checkpoint_averaging.py ${CKPT_DIR}

NEMO_MODEL="${CKPT_DIR}/Conformer-Transformer-SLU2ASR-averaged.nemo"
CUDA_VISIBLE_DEVICES=1 python run_slurp_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu2asr.json" \
    model_path=${NEMO_MODEL} \
    batch_size=16 \
    num_workers=8 \
    searcher.type="greedy" \
    searcher.beam_size=32 \
    searcher.temperature=1.25 \
    only_score_manifest=false


# dataset_manifest="${DATA_DIR}/test_slu2asr.json"
# dataset_manifest="evaluation_transcripts.json"
