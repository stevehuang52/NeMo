curr_dir=${pwd}

# proj_name="ASR_Finetune"
# exp_dir="drc_ConfLCTC_SLURP_adamwlr1e-4_wd1e-3_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep50_r2_dgx1"
# proj_name="SLURP_SLU2ASR"
# exp_dir="drc_ConformerL-Transformer-Adapter_decl8_adp64_adamwlr3e-4x3e-4_wd0.0_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep100"

proj_name="ConformerL_ctc_slurp"
exp_dir="drc_SLURP_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep600_asr_dgx1"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

python checkpoint_averaging.py ${target_dir}
