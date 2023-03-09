
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
exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"  # best
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_fnorm_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_trns_wce_n8"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns2_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n8_r2"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-3minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_n4"

# exp_dir="drc_marblenet_c68_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_ns_wce_synth_s2_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth_wce_ep50_s2_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_wce_ep50_n4"
# exp_dir="drc_marblenet_c68_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s4_wce_ep50_n4"
# exp_dir="drc_marblenet_c68_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4_r2"

# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_SynthMulti_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_multisynth_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_ga_wce_ep50_n4"


# "/media/data2/simulated_data/ls960_dur20_spks2_sln0.4_ovl0.1_seed42_noisy_100h.json"

# ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"
ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

model_path="${ckpt_dir}/${exp_dir}-averaged.nemo"
# data_dir=./manifests_test
data_dir=/media/data2/simulated_data/manifests

CUDA_VISIBLE_DEVICES=1 python infer_vad_multi.py \
    --config-path="./configs" --config-name="vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="${ckpt_dir}/frame_vad_multi_output_noisy" \
    dataset="[${data_dir}/vox1_dur180_spks3_turnP0.85_ovl0.15x0.01_sln0.3x0.01_seed9_test_snr5_gap0.5m25m15_100h.json]"
    # dataset="[${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.3_seed9_snr0_test_100h.json,${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.7_seed6_snr0_test_100h.json,${data_dir}/ls960_dur180_spk3_ovl0.15_sln0.7_seed42_test-clean_noisy_snr0_100h.json]"

    # dataset="[${data_dir}/ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h.json,${data_dir}/vox1_dur60_spk2_ovl0.15_sln0.5_seed777_test_snr0_100h.json]"
    
    # dataset="[${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.3_seed9_snr0_test_100h.json,${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.7_seed6_snr0_test_100h.json,${data_dir}/ls960_dur180_spk3_ovl0.15_sln0.7_seed42_test-clean_noisy_snr0_100h.json]"



    # dataset="[${data_dir}/ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h.json]"
    # dataset="[${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.3_seed9_snr0_test_100h.json,${data_dir}/vox1_dur180_spk3_ovl0.15_sln0.7_seed6_snr0_test_100h.json,${data_dir}/ls960_dur180_spk3_ovl0.15_sln0.7_seed42_test-clean_noisy_snr0_100h.json]"

# ${data_dir}/ls960_dur20_spks2_ovl0.1_sln0.4_seed1_test_noisy_100h.json
    # dataset="[${data_dir}/french_test_20ms.json,${data_dir}/german_test_20ms.json,${data_dir}/mandarin_test_20ms.json,${data_dir}/russian_test_20ms.json,${data_dir}/spanish_test_20ms.json]"



# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50" 


# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50" 
# [NeMo I 2023-02-15 20:04:58 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-15 20:04:58 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-15 20:06:25 infer_vad_multi:116] ================= vox1_dur60_spk2_ovl0.15_sln0.5_seed777_test_snr0_100h =================
# [NeMo I 2023-02-15 20:06:25 infer_vad_multi:117] AUROC: 0.9277
# [NeMo I 2023-02-15 20:06:25 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 20:06:25 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.77      0.84      0.80  13425574
#                1       0.90      0.86      0.88  23550830
    
#         accuracy                           0.85  36976404
#        macro avg       0.84      0.85      0.84  36976404
#     weighted avg       0.86      0.85      0.85  36976404
    
# [NeMo I 2023-02-15 20:06:25 infer_vad_multi:125] Detection Error Rate: DetER=22.2470, False Alarm=14.1671, Miss=8.0799


# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_wce_ep50_n4"
# [NeMo I 2023-02-15 18:11:57 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-15 18:11:57 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-15 18:13:30 infer_vad_multi:116] ================= vox1_dur60_spk2_ovl0.15_sln0.5_seed777_test_snr0_100h =================
# [NeMo I 2023-02-15 18:13:30 infer_vad_multi:117] AUROC: 0.9533
# [NeMo I 2023-02-15 18:13:30 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 18:13:30 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.68      0.96      0.80  13425574
#                1       0.97      0.75      0.84  23550830
    
#         accuracy                           0.82  36976404
#        macro avg       0.83      0.85      0.82  36976404
#     weighted avg       0.87      0.82      0.83  36976404
    

# [NeMo I 2023-02-15 18:13:30 infer_vad_multi:125] Detection Error Rate: DetER=22.1083, False Alarm=4.6065, Miss=17.5019
# [NeMo I 2023-0

# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_wce_ep50_n4"
# NeMo I 2023-02-15 16:34:59 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-15 16:34:59 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-15 16:35:48 infer_vad_multi:116] ================= ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h =================
# [NeMo I 2023-02-15 16:35:48 infer_vad_multi:117] AUROC: 0.9705
# [NeMo I 2023-02-15 16:35:48 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 16:35:48 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.75      0.93      0.83   4937088
#                1       0.97      0.90      0.93  14582004
    
#         accuracy                           0.90  19519092
#        macro avg       0.86      0.91      0.88  19519092
#     weighted avg       0.92      0.90      0.91  19519092
    

# exp_dir="drc_marblenet_c68_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_ns_wce_synth_s2_n4"
# [NeMo I 2023-02-15 16:15:12 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-15 16:15:12 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-15 16:15:48 infer_vad_multi:116] ================= ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h =================
# [NeMo I 2023-02-15 16:15:48 infer_vad_multi:117] AUROC: 0.9667
# [NeMo I 2023-02-15 16:15:48 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 16:15:48 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.73      0.93      0.82   2468544
#                1       0.97      0.88      0.93   7291002
    
#         accuracy                           0.89   9759546
#        macro avg       0.85      0.91      0.87   9759546
#     weighted avg       0.91      0.89      0.90   9759546
    

