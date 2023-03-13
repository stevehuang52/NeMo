
# proj_name="Stream_VAD"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"
# ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

# exp_dir="marblenet_3x2x64_mandarin_40ms_all"
# ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"

proj_name="Frame_VAD"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b512_gacc1_ep100_ns_w8_sfish"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep100_sf_ns_wce_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_fnorm_ep50"
exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"

ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

# "/media/data2/simulated_data/ls960_dur20_spks2_sln0.4_ovl0.1_seed42_noisy_100h.json"
# split="ami_dev_10ms"
# split="ch120_moved_10ms"

split="dh3_dev_all_excts"

output_dir="${ckpt_dir}/frame_vad_multi_output_sd_dev/vad_output_${split}"
pred_dir="${output_dir}/frames_predictions"
gt_dir="${output_dir}/frames_groundtruth"

python run_grid_search.py \
    pred_dir=$pred_dir \
    gt_dir=$gt_dir
