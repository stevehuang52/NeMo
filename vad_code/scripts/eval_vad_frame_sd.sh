
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

# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_wce_ep50_n4"
# exp_dir="drc_marblenet_c68_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s4_wce_ep50_n4"

# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"  # best
# exp_dir="drc_marblenet_c68_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4_r2"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"

exp_dir="drc_marblenet_3x2x64_Mixed_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_SynthMulti_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_multisynth_s2_wce_ep50_n4"
# exp_dir="drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1k_s2_ga_wce_ep50_n4"


# "/media/data2/simulated_data/ls960_dur20_spks2_sln0.4_ovl0.1_seed42_noisy_100h.json"

# ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"
ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

model_path="${ckpt_dir}/${exp_dir}-averaged.nemo"
data_dir=/media/data/projects/NeMo-fvad/vad_code/manifests_sd_eval_40ms

CUDA_VISIBLE_DEVICES=0 python infer_vad_multi.py \
    --config-path="./configs" --config-name="vad_inference" \
    vad.model_path=$model_path \
    vad.parameters.shift_length_in_sec=0.02 \
    frame_out_dir="${ckpt_dir}/frame_vad_multi_output_sd_dev" \
    dataset="[${data_dir}/dh3_dev_all_excts.json]"
    # dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json,${data_dir}/dh3_dev_clinical_manifest.json,${data_dir}/dh3_dev_court_manifest.json,${data_dir}/dh3_dev_cts_manifest.json,${data_dir}/dh3_dev_maptask_manifest.json,${data_dir}/dh3_dev_meeting_manifest.json,${data_dir}/dh3_dev_restaurant_manifest.json,${data_dir}/dh3_dev_socio_field_manifest.json,${data_dir}/dh3_dev_socio_lab_manifest.json,${data_dir}/dh3_dev_webvideo_manifest.json]"

    # dataset="[${data_dir}/voxconv_test_full_manifest.json]"
    # dataset="[${data_dir}/dh3_eval_audiobooks_manifest.json,${data_dir}/dh3_eval_broadcast_interview_manifest.json,${data_dir}/dh3_eval_clinical_manifest.json,${data_dir}/dh3_eval_court_manifest.json,${data_dir}/dh3_eval_cts_manifest.json,${data_dir}/dh3_eval_maptask_manifest.json,${data_dir}/dh3_eval_meeting_manifest.json,${data_dir}/dh3_eval_restaurant_manifest.json,${data_dir}/dh3_eval_socio_field_manifest.json,${data_dir}/dh3_eval_socio_lab_manifest.json,${data_dir}/dh3_eval_webvideo_manifest.json]"
    # dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json,${data_dir}/dh3_dev_clinical_manifest.json,${data_dir}/dh3_dev_court_manifest.json,${data_dir}/dh3_dev_cts_manifest.json,${data_dir}/dh3_dev_maptask_manifest.json,${data_dir}/dh3_dev_meeting_manifest.json,${data_dir}/dh3_dev_restaurant_manifest.json,${data_dir}/dh3_dev_socio_field_manifest.json,${data_dir}/dh3_dev_socio_lab_manifest.json,${data_dir}/dh3_dev_webvideo_manifest.json]"


# NeMo I 2023-02-20 15:13:54 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-20 15:13:54 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:116] ================= dh3_eval_audiobooks_manifest =================
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:117] AUROC: 0.9612
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.89      0.73      0.80     81216
#                1       0.93      0.97      0.95    286172
    
#         accuracy                           0.92    367388
#        macro avg       0.91      0.85      0.88    367388
#     weighted avg       0.92      0.92      0.92    367388
    
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:125] Detection Error Rate: DetER=18.6564, False Alarm=17.7697, Miss=0.8867
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:116] ================= dh3_eval_broadcast_interview_manifest =================
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:117] AUROC: 0.9323
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.75      0.72      0.73     82356
#                1       0.92      0.93      0.92    282784
    
#         accuracy                           0.88    365140
#        macro avg       0.83      0.82      0.83    365140
#     weighted avg       0.88      0.88      0.88    365140
    
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:125] Detection Error Rate: DetER=20.4418, False Alarm=17.0681, Miss=3.3737
# [NeMo I 2023-02-20 15:13:55 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:116] ================= dh3_eval_clinical_manifest =================
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:117] AUROC: 0.8744
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.57      0.92      0.70    297786
#                1       0.92      0.58      0.71    486221
    
#         accuracy                           0.71    784007
#        macro avg       0.75      0.75      0.71    784007
#     weighted avg       0.79      0.71      0.71    784007
    
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:125] Detection Error Rate: DetER=42.2371, False Alarm=10.6123, Miss=31.6248
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:116] ================= dh3_eval_court_manifest =================
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:117] AUROC: 0.9433
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.93      0.39      0.55     61870
#                1       0.89      0.99      0.94    305615
    
#         accuracy                           0.89    367485
#        macro avg       0.91      0.69      0.75    367485
#     weighted avg       0.90      0.89      0.87    367485
    
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:125] Detection Error Rate: DetER=15.8472, False Alarm=15.7339, Miss=0.1133
# [NeMo I 2023-02-20 15:13:57 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:116] ================= dh3_eval_cts_manifest =================
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:117] AUROC: 0.9009
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.77      0.34      0.47    195915
#                1       0.93      0.99      0.96   1634025
    
#         accuracy                           0.92   1829940
#        macro avg       0.85      0.66      0.71   1829940
#     weighted avg       0.91      0.92      0.90   1829940
    
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:125] Detection Error Rate: DetER=9.8838, False Alarm=9.5183, Miss=0.3655
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:116] ================= dh3_eval_maptask_manifest =================
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:117] AUROC: 0.9623
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.87      0.86      0.87    131892
#                1       0.93      0.93      0.93    241591
    
#         accuracy                           0.91    373483
#        macro avg       0.90      0.90      0.90    373483
#     weighted avg       0.91      0.91      0.91    373483
    
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:125] Detection Error Rate: DetER=26.0331, False Alarm=23.5882, Miss=2.4449
# [NeMo I 2023-02-20 15:14:01 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:116] ================= dh3_eval_meeting_manifest =================
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:117] AUROC: 0.8086
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.19      0.99      0.32     59833
#                1       0.98      0.08      0.15    276813
    
#         accuracy                           0.25    336646
#        macro avg       0.59      0.54      0.24    336646
#     weighted avg       0.84      0.25      0.18    336646
    
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:125] Detection Error Rate: DetER=86.2305, False Alarm=0.3618, Miss=85.8687
# [NeMo I 2023-02-20 15:14:02 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:116] ================= dh3_eval_restaurant_manifest =================
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:117] AUROC: 0.8523
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.17      0.98      0.29     43998
#                1       0.99      0.34      0.51    326306
    
#         accuracy                           0.42    370304
#        macro avg       0.58      0.66      0.40    370304
#     weighted avg       0.89      0.42      0.48    370304
    
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:125] Detection Error Rate: DetER=55.8841, False Alarm=0.5339, Miss=55.3502
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:116] ================= dh3_eval_socio_field_manifest =================
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:117] AUROC: 0.8829
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.55      0.77      0.64     91899
#                1       0.92      0.82      0.87    316301
    
#         accuracy                           0.81    408200
#        macro avg       0.74      0.79      0.75    408200
#     weighted avg       0.84      0.81      0.82    408200
    
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:125] Detection Error Rate: DetER=23.2939, False Alarm=11.9240, Miss=11.3698
# [NeMo I 2023-02-20 15:14:03 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:116] ================= dh3_eval_socio_lab_manifest =================
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:117] AUROC: 0.8406
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.36      0.94      0.52     87634
#                1       0.96      0.47      0.63    278030
    
#         accuracy                           0.58    365664
#        macro avg       0.66      0.70      0.57    365664
#     weighted avg       0.82      0.58      0.60    365664
    
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:125] Detection Error Rate: DetER=46.7267, False Alarm=5.9389, Miss=40.7878
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:116] ================= dh3_eval_webvideo_manifest =================
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:117] AUROC: 0.7949
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.42      0.83      0.56    100162
#                1       0.90      0.57      0.70    273263
    
#         accuracy                           0.64    373425
#        macro avg       0.66      0.70      0.63    373425
#     weighted avg       0.77      0.64      0.66    373425
    
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:125] Detection Error Rate: DetER=43.7549, False Alarm=10.2841, Miss=33.4708
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:128] ================== Aggregrated Results ===================
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:132] ============================================================
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:133]  DetER=43.7549, False Alarm=10.2841, Miss=33.4708
# [NeMo I 2023-02-20 15:14:04 infer_vad_multi:134] ============================================================
# [NeMo I 2023-02-20 15:14:18 infer_vad_multi:139] AUROC: 0.8520
# [NeMo I 2023-02-20 15:14:18 infer_vad_multi:140] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:14:18 infer_vad_multi:141] 
#                   precision    recall  f1-score   support
    
#                0       0.47      0.76      0.58   1234561
#                1       0.92      0.77      0.84   4707121
    
#         accuracy                           0.77   5941682
#        macro avg       0.70      0.77      0.71   5941682
#     weighted avg       0.83      0.77      0.79   5941682
    
# [NeMo I 2023-02-20 15:14:18 infer_vad_multi:143] {'onset': 0.5, 'offset': 0.4, 'pad_onset': 0.1, 'pad_offset': 0.1, 'min_duration_on': 0.1, 'min_duration_off': 0.1, 'filter_speech_first': True}
# [NeMo I 2023-02-20 15:14:18 infer_vad_multi:144] Done.
# /media/data/projects/NeMo-fvad/vad_code/nemo_experiments/Frame_VAD/drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4/drc_marblenet_3x2x64_Synth_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b64_gacc1_ep50_synth1500only_s2_wce_ep50_n4-averaged.nemo


# [NeMo I 2023-02-20 15:45:12 infer_vad_multi:100] ====================================================
# [NeMo I 2023-02-20 15:45:12 infer_vad_multi:101] Finalizing individual results...
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:116] ================= voxconv_test_full_manifest =================
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:117] AUROC: 0.9468
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:119]               precision    recall  f1-score   support
    
#                0       0.57      0.80      0.67    810906
#                1       0.98      0.93      0.95   7026053
    
#         accuracy                           0.92   7836959
#        macro avg       0.77      0.87      0.81   7836959
#     weighted avg       0.93      0.92      0.92   7836959
    
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:125] Detection Error Rate: DetER=7.0378, False Alarm=3.9904, Miss=3.0474
# [NeMo I 2023-02-20 15:45:30 infer_vad_multi:126] ==========================================
    



