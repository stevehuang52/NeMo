#!/bin/bash
curr_dir=${pwd}

# NEMO_BASEPATH="/media/data2/nemo-simulator2"
# echo $NEMO_BASEPATH
# export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

proj_name="Frame_VAD"
# exp_dir="drc_Multilang_sgdlr1e-2_wd1e-3_augx_b512_gacc1_ep50_ns_wce_td_n4"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b512_gacc1_ep100_ns_w8_sfish"
# exp_dir="drc_conformer_small_Multilang_adamwlr5.0_wd1e-3_aug5x0.05_b32_gacc1_ep50_ns_wce"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep100_sf_ns_wce_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_fnorm_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_trns_wce_n8"
# exp_dir="drc_conformer_small_Multilang_adamwlr5.0_wd1e-3_aug5x0.05_b32_gacc1_ep50_trns_wce_fnorm"
# exp_dir="drc_conformer_tiny_chunck_Multilang_l3_adamwlr3.0_wd1e-3_aug5x0.05_b64_gacc1_ep50_ns_wce_fnorm_n8"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns2_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n8_r2"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-3minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n4"

# exp_dir="drc_marblenet_c68_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_ns_wce_synth_s2_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_s2_n4"
exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_wce_ep50_n4"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco:${source_dir} ${target_dir}

python checkpoint_averaging.py ${target_dir}
