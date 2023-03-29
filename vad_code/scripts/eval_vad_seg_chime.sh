


# data_dir=./manifests_test
data_dir=/media/data2/simulated_data/manifests
out_dir=/media/data/projects/NeMo-fvad/vad_code/nemo_experiments
CUDA_VISIBLE_DEVICES=1 python infer_seg_vad.py \
    --config-path="./configs" --config-name="vad_seg_inference" \
    frame_out_dir="${out_dir}/segment_vad_multi_output" \
    dataset="[/media/data2/chime7-challenge/datasets/manifests_task1/mixer6/mulspk_asr_manifest/mixer6-dev.mc_split.json]"    
#     dataset="[/media/data2/chime7-challenge/datasets/manifests_task1/mixer6/mulspk_asr_manifest/mixer6-debug.json,/media/data2/chime7-challenge/datasets/manifests_task1/mixer6/mulspk_asr_manifest/mixer6-debug2.json]"    
    # dataset="[${data_dir}/french_test_20ms.json,${data_dir}/german_test_20ms.json,${data_dir}/mandarin_test_20ms.json,${data_dir}/russian_test_20ms.json,${data_dir}/spanish_test_20ms.json]"





#     dataset="[${data_dir}/vox1_dur60_spk2_ovl0.15_sln0.5_seed777_test_snr0_100h.json]"
# [NeMo I 2023-02-15 18:12:19 infer_seg_vad:100] ====================================================
# [NeMo I 2023-02-15 18:12:19 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-15 18:12:29 infer_seg_vad:116] ================= vox1_dur60_spk2_ovl0.15_sln0.5_seed777_test_snr0_100h =================
# [NeMo I 2023-02-15 18:12:29 infer_seg_vad:117] AUROC: 0.9337
# [NeMo I 2023-02-15 18:12:29 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 18:12:29 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.67      0.96      0.79   1668913
#                1       0.97      0.74      0.84   2950668
    
#         accuracy                           0.82   4619581
#        macro avg       0.82      0.85      0.81   4619581
#     weighted avg       0.86      0.82      0.82   4619581
    
# [NeMo I 2023-02-15 18:12:29 infer_seg_vad:125] Detection Error Rate: DetER=19.7579, False Alarm=8.4779, Miss=11.2800

# dataset="[${data_dir}/ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h.json]"
# NeMo I 2023-02-15 16:39:30 infer_seg_vad:100] ====================================================
# [NeMo I 2023-02-15 16:39:30 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-15 16:39:35 infer_seg_vad:116] ================= ls960_spk2_ovl0.1_sln0.4_seed2_test-clean_noisy_snr0_50h =================
# [NeMo I 2023-02-15 16:39:35 infer_seg_vad:117] AUROC: 0.9488
# [NeMo I 2023-02-15 16:39:35 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 16:39:35 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.70      0.89      0.78    606795
#                1       0.96      0.87      0.91   1829399
    
#         accuracy                           0.88   2436194
#        macro avg       0.83      0.88      0.85   2436194
#     weighted avg       0.90      0.88      0.88   2436194


#  dataset="[${data_dir}/ls960_dur20_spks2_ovl0.1_sln0.4_seed1_dev_noisy_100h.json,${data_dir}/ls960_dur20_spks2_ovl0.1_sln0.4_seed1_test_noisy_100h.json]" \
#     [NeMo I 2023-02-15 14:01:47 infer_seg_vad:100] ====================================================
# [NeMo I 2023-02-15 14:01:47 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:116] ================= ls960_dur20_spks2_ovl0.1_sln0.4_seed1_dev_noisy_100h =================
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:117] AUROC: 0.9764
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.90      0.86      0.88   1204268
#                1       0.95      0.97      0.96   3594683
    
#         accuracy                           0.94   4798951
#        macro avg       0.93      0.92      0.92   4798951
#     weighted avg       0.94      0.94      0.94   4798951
    
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:125] Detection Error Rate: DetER=16.0039, False Alarm=15.4036, Miss=0.6003
# [NeMo I 2023-02-15 14:01:59 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:116] ================= ls960_dur20_spks2_ovl0.1_sln0.4_seed1_test_noisy_100h =================
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:117] AUROC: 0.9782
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.92      0.86      0.89   1203282
#                1       0.95      0.98      0.96   3646777
    
#         accuracy                           0.95   4850059
#        macro avg       0.94      0.92      0.93   4850059
#     weighted avg       0.95      0.95      0.95   4850059
    
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:125] Detection Error Rate: DetER=15.4011, False Alarm=15.2753, Miss=0.1258
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:128] ================== Aggregrated Results ===================
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:132] ============================================================
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:133]  DetER=15.4011, False Alarm=15.2753, Miss=0.1258
# [NeMo I 2023-02-15 14:02:12 infer_seg_vad:134] ============================================================
# [NeMo I 2023-02-15 14:03:20 infer_seg_vad:139] AUROC: 0.9773
# [NeMo I 2023-02-15 14:03:20 infer_seg_vad:140] Classification report with threshold=0.50
# [NeMo I 2023-02-15 14:03:20 infer_seg_vad:141] 
#                   precision    recall  f1-score   support
    
#                0       0.91      0.86      0.88   2407550
#                1       0.95      0.97      0.96   7241460
    
#         accuracy                           0.94   9649010
#        macro avg       0.93      0.92      0.92   9649010
#     weighted avg       0.94      0.94      0.94   9649010
    