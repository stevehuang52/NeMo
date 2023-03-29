
# proj_name="Stream_VAD"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"
# ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

# exp_dir="marblenet_3x2x64_mandarin_40ms_all"
# exp_dir="marblenet_3x2x64_multilang_40ms_all_noise"
proj_name="Frame_VAD"

# exp_dir="drc_conformer_small_Multilang_adamwlr5.0_wd1e-3_aug5x0.05_b32_gacc1_ep50_ns_wce"
# exp_dir="drc_conformer_tiny_chunck_Multilang_l3_adamwlr3.0_wd1e-3_aug5x0.05_b64_gacc1_ep50_ns_wce_fnorm_n8"
# exp_dir="drc_conformer_small_Multilang_adamwlr5.0_wd1e-3_aug5x0.05_b32_gacc1_ep50_trns_wce_fnorm"

# exp_dir="drc_Multilang_sgdlr1e-2_wd1e-3_augx_b512_gacc1_ep50_ns_wce_td_n4"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b512_gacc1_ep100_ns_w8_sfish"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep100_sf_ns_wce_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"  # best
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_fnorm_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_trns_wce_n8"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns2_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n8_r2"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-3minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n4"

# exp_dir="drc_marblenet_c68_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_ns_wce_synth_s2_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_s2_n4"
# exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_SynthMulti_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_multisynth_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_ep50_synth1k_s2_wce_fnorm_n4"
exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_ep50_synth1k_s2_wce_gain20_n4"


# ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"
ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

model_path="${ckpt_dir}/${exp_dir}-averaged.nemo"
data_dir=./manifests_test

CUDA_VISIBLE_DEVICES=0 python infer_vad_multi.py \
    --config-path="./configs" --config-name="vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="${ckpt_dir}/frame_vad_multi_output" \
    dataset="[${data_dir}/ava_eval_10ms.json,${data_dir}/ami_eval_10ms.json,${data_dir}/ch120_CH109_10ms.json]"

    # dataset="[${data_dir}/french_test_20ms.json,${data_dir}/german_test_20ms.json,${data_dir}/mandarin_test_20ms.json,${data_dir}/russian_test_20ms.json,${data_dir}/spanish_test_20ms.json]"