
DATA_DIR="/home/heh/datasets/slurp_data"
EXP_NAME="slurp_conformer_transformer_large_fp16"
TIMESTAMP="2022-08-29_13-58-13"
CKPT_DIR="nemo_experiments/${EXP_NAME}/${TIMESTAMP}/checkpoints"

python ../../../scripts/checkpoint_averaging/checkpoint_averaging.py ../../../examples/slu/speech_intent_slot/${CKPT_DIR}

NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"
CUDA_VISIBLE_DEVICES=0 python run_speech_intent_slot_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu.json" \
    model_path=${NEMO_MODEL} \
    batch_size=32 \
    num_workers=8 \
    sequence_generator.type="beam" \
    sequence_generator.beam_size=32 \
    sequence_generator.temperature=1.25 \
    only_score_manifest=false
