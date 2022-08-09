DATA_DIR="/home/heh/datasets/slurp_draco"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/2022-06-29_10-41-00/checkpoints"
# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/Conformer-Transformer-SLU2ASR/checkpoints"

# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_2LR_ep200"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d512h4"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_adapter"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4_dec3_d2048h8_freezeEnc"
# EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8_r2"

# CKPT_DIR="/home/heh/github/NeMo/codes/nemo_experiments/${EXP_NAME}/checkpoints"

proj_name="SLURP_SLU2ASR"
# exp_dir="drc_ConformerL-Transformer-Adapter_decl8_adp64_adamwlr3e-4x3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep100"
# exp_dir="drc_ConformerL-Transformer_dl3_adamwlr3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep100_nossl"
# exp_dir="drc_ConformerL-Transformer_dl3_adamwlr3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep100_ftasr"
# exp_dir="drc_ConformerXL-Transformer_dl3_adamwlr3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b4_ep50_dgx1"
exp_dir="drc_ConformerL-Transformer_dl3_adamwlr3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep100_ftasr_frzenc"


proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/
checkpoint_dir=${target_dir}/checkpoints

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

# echo $checkpoint_dir

python checkpoint_averaging.py $checkpoint_dir

NEMO_MODEL="${checkpoint_dir}/${exp_dir}-averaged.nemo"
CUDA_VISIBLE_DEVICES=1 python run_slurp_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu2asr.json" \
    model_path=${NEMO_MODEL} \
    batch_size=16 \
    num_workers=8 \
    searcher.type="beam" \
    searcher.beam_size=32 \
    searcher.temperature=1.25 \
    only_score_manifest=false

# dataset_manifest="${target_dir}/predictions.json"
# dataset_manifest="${DATA_DIR}/test_slu2asr.json"
