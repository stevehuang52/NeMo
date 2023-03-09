
data_dir=/media/data/projects/NeMo-fvad/vad_code/manifests_sd_eval_40ms
out_dir=/media/data/projects/NeMo-fvad/vad_code/nemo_experiments
CUDA_VISIBLE_DEVICES=0 python infer_seg_vad.py \
    --config-path="./configs" --config-name="vad_seg_inference" \
    frame_out_dir="${out_dir}/segment_vad_multi_output" \
    vad.parameters.shift_length_in_sec=0.08 \
    dataset="[${data_dir}/dh3_dev_audiobooks_manifest.json,${data_dir}/dh3_dev_broadcast_interview_manifest.json,${data_dir}/dh3_dev_clinical_manifest.json,${data_dir}/dh3_dev_court_manifest.json,${data_dir}/dh3_dev_cts_manifest.json,${data_dir}/dh3_dev_maptask_manifest.json,${data_dir}/dh3_dev_meeting_manifest.json,${data_dir}/dh3_dev_restaurant_manifest.json,${data_dir}/dh3_dev_socio_field_manifest.json,${data_dir}/dh3_dev_socio_lab_manifest.json,${data_dir}/dh3_dev_webvideo_manifest.json]"

    # dataset="[${data_dir}/voxconv_test_full_manifest.json]"
    # dataset="[${data_dir}/dh3_eval_audiobooks_manifest.json,${data_dir}/dh3_eval_broadcast_interview_manifest.json,${data_dir}/dh3_eval_clinical_manifest.json,${data_dir}/dh3_eval_court_manifest.json,${data_dir}/dh3_eval_cts_manifest.json,${data_dir}/dh3_eval_maptask_manifest.json,${data_dir}/dh3_eval_meeting_manifest.json,${data_dir}/dh3_eval_restaurant_manifest.json,${data_dir}/dh3_eval_socio_field_manifest.json,${data_dir}/dh3_eval_socio_lab_manifest.json,${data_dir}/dh3_eval_webvideo_manifest.json]"

#     dataset="[${data_dir}/voxconv_test_full_manifest.json]"

# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:100] ====================================================
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:116] ================= dh3_dev_audiobooks_manifest =================
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:117] AUROC: 0.8915
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.94      0.35      0.51     17267
#                1       0.87      0.99      0.93     73274
    
#         accuracy                           0.87     90541
#        macro avg       0.90      0.67      0.72     90541
#     weighted avg       0.88      0.87      0.85     90541
    
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:125] Detection Error Rate: DetER=21.4093, False Alarm=21.3793, Miss=0.0301
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:116] ================= dh3_dev_broadcast_interview_manifest =================
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:117] AUROC: 0.9130
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.95      0.34      0.50     18475
#                1       0.86      1.00      0.92     74231
    
#         accuracy                           0.86     92706
#        macro avg       0.91      0.67      0.71     92706
#     weighted avg       0.88      0.86      0.84     92706
    
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:125] Detection Error Rate: DetER=23.3136, False Alarm=23.2618, Miss=0.0518
# [NeMo I 2023-02-21 15:13:42 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:116] ================= dh3_dev_clinical_manifest =================
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:117] AUROC: 0.8449
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.62      0.83      0.71     73183
#                1       0.87      0.69      0.77    118851
    
#         accuracy                           0.74    192034
#        macro avg       0.74      0.76      0.74    192034
#     weighted avg       0.77      0.74      0.75    192034
    
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:125] Detection Error Rate: DetER=39.4094, False Alarm=27.6841, Miss=11.7253
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:116] ================= dh3_dev_court_manifest =================
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:117] AUROC: 0.9016
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.97      0.23      0.38     13707
#                1       0.88      1.00      0.94     79989
    
#         accuracy                           0.89     93696
#        macro avg       0.93      0.62      0.66     93696
#     weighted avg       0.90      0.89      0.86     93696
    
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:125] Detection Error Rate: DetER=15.8172, False Alarm=15.8116, Miss=0.0056
# [NeMo I 2023-02-21 15:13:43 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:116] ================= dh3_dev_cts_manifest =================
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:117] AUROC: 0.7349
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.85      0.26      0.40     42619
#                1       0.93      1.00      0.96    414881
    
#         accuracy                           0.93    457500
#        macro avg       0.89      0.63      0.68    457500
#     weighted avg       0.92      0.93      0.91    457500
    
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:125] Detection Error Rate: DetER=9.6308, False Alarm=9.5916, Miss=0.0392
# [NeMo I 2023-02-21 15:13:44 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:116] ================= dh3_dev_maptask_manifest =================
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:117] AUROC: 0.8874
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.97      0.50      0.66     33957
#                1       0.82      0.99      0.90     79697
    
#         accuracy                           0.85    113654
#        macro avg       0.90      0.75      0.78    113654
#     weighted avg       0.87      0.85      0.83    113654
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:125] Detection Error Rate: DetER=34.0351, False Alarm=33.9396, Miss=0.0955
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:116] ================= dh3_dev_meeting_manifest =================
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:117] AUROC: 0.8455
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.17      0.78      0.28      6464
#                1       0.98      0.76      0.86    103689
    
#         accuracy                           0.76    110153
#        macro avg       0.58      0.77      0.57    110153
#     weighted avg       0.93      0.76      0.82    110153
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:125] Detection Error Rate: DetER=13.3991, False Alarm=3.6065, Miss=9.7926
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:116] ================= dh3_dev_restaurant_manifest =================
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:117] AUROC: 0.8591
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.17      0.96      0.29     10545
#                1       0.99      0.39      0.56     80626
    
#         accuracy                           0.46     91171
#        macro avg       0.58      0.68      0.43     91171
#     weighted avg       0.89      0.46      0.53     91171
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:125] Detection Error Rate: DetER=34.2796, False Alarm=2.7747, Miss=31.5049
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:116] ================= dh3_dev_socio_field_manifest =================
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:117] AUROC: 0.8459
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.68      0.55      0.61     23076
#                1       0.86      0.91      0.88     67475
    
#         accuracy                           0.82     90551
#        macro avg       0.77      0.73      0.74     90551
#     weighted avg       0.81      0.82      0.81     90551
    
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:125] Detection Error Rate: DetER=29.0375, False Alarm=26.2576, Miss=2.7798
# [NeMo I 2023-02-21 15:13:45 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:116] ================= dh3_dev_socio_lab_manifest =================
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:117] AUROC: 0.8491
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.55      0.71      0.62     28615
#                1       0.90      0.82      0.86     91382
    
#         accuracy                           0.79    119997
#        macro avg       0.72      0.76      0.74    119997
#     weighted avg       0.82      0.79      0.80    119997
    
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:125] Detection Error Rate: DetER=25.8322, False Alarm=21.2807, Miss=4.5515
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:116] ================= dh3_dev_webvideo_manifest =================
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:117] AUROC: 0.8250
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.42      0.86      0.56     20487
#                1       0.93      0.62      0.74     64381
    
#         accuracy                           0.68     84868
#        macro avg       0.68      0.74      0.65     84868
#     weighted avg       0.81      0.68      0.70     84868
    
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:125] Detection Error Rate: DetER=30.3214, False Alarm=12.3436, Miss=17.9778
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:128] ================== Aggregrated Results ===================
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:132] ============================================================
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:133]  DetER=30.3214, False Alarm=12.3436, Miss=17.9778
# [NeMo I 2023-02-21 15:13:46 infer_seg_vad:134] ============================================================
# [NeMo I 2023-02-21 15:13:50 infer_seg_vad:139] AUROC: 0.7938
# [NeMo I 2023-02-21 15:13:50 infer_seg_vad:140] Classification report with threshold=0.50
# [NeMo I 2023-02-21 15:13:50 infer_seg_vad:141] 
#                   precision    recall  f1-score   support
    
#                0       0.51      0.59      0.55    288395
#                1       0.90      0.87      0.89   1248476
    
#         accuracy                           0.82   1536871
#        macro avg       0.71      0.73      0.72   1536871
#     weighted avg       0.83      0.82      0.82   1536871
    
# [NeMo I 2023-02-21 15:13:50 infer_seg_vad:143] {'onset': 0.5, 'offset': 0.2, 'pad_onset': 0.3, 'pad_offset': 0.3, 'min_duration_on': 0.1, 'min_duration_off': 0.1, 'filter_speech_first': True}
# [NeMo I 2023-02-21 15:13:50 infer_seg_vad:144] Done.


# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:100] ====================================================
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:116] ================= dh3_eval_audiobooks_manifest =================
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:117] AUROC: 0.8947
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.93      0.34      0.50     19008
#                1       0.85      0.99      0.92     72839
    
#         accuracy                           0.86     91847
#        macro avg       0.89      0.67      0.71     91847
#     weighted avg       0.87      0.86      0.83     91847
    
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:125] Detection Error Rate: DetER=23.7585, False Alarm=23.5938, Miss=0.1647
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:116] ================= dh3_eval_broadcast_interview_manifest =================
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:117] AUROC: 0.8901
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.87      0.40      0.55     19226
#                1       0.86      0.98      0.92     72059
    
#         accuracy                           0.86     91285
#        macro avg       0.86      0.69      0.73     91285
#     weighted avg       0.86      0.86      0.84     91285
    
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:125] Detection Error Rate: DetER=24.0232, False Alarm=23.8920, Miss=0.1312
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:116] ================= dh3_eval_clinical_manifest =================
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:117] AUROC: 0.8744
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.68      0.80      0.73     71046
#                1       0.87      0.78      0.82    124925
    
#         accuracy                           0.79    195971
#        macro avg       0.77      0.79      0.78    195971
#     weighted avg       0.80      0.79      0.79    195971
    
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:125] Detection Error Rate: DetER=35.4839, False Alarm=27.3325, Miss=8.1514
# [NeMo I 2023-02-20 15:31:01 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:116] ================= dh3_eval_court_manifest =================
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:117] AUROC: 0.8996
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.98      0.23      0.37     14188
#                1       0.88      1.00      0.93     77684
    
#         accuracy                           0.88     91872
#        macro avg       0.93      0.61      0.65     91872
#     weighted avg       0.89      0.88      0.85     91872
    
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:125] Detection Error Rate: DetER=16.8788, False Alarm=16.8697, Miss=0.0091
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:116] ================= dh3_eval_cts_manifest =================
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:117] AUROC: 0.6997
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.87      0.24      0.38     43966
#                1       0.93      1.00      0.96    413534
    
#         accuracy                           0.92    457500
#        macro avg       0.90      0.62      0.67    457500
#     weighted avg       0.92      0.92      0.90    457500
    
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:125] Detection Error Rate: DetER=10.0442, False Alarm=10.0264, Miss=0.0178
# [NeMo I 2023-02-20 15:31:02 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_maptask_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.8986
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.98      0.51      0.67     31063
#                1       0.80      0.99      0.89     62298
    
#         accuracy                           0.83     93361
#        macro avg       0.89      0.75      0.78     93361
#     weighted avg       0.86      0.83      0.81     93361
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=38.7534, False Alarm=38.6355, Miss=0.1179
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_meeting_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.7673
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.26      0.82      0.39     13778
#                1       0.94      0.54      0.69     70381
    
#         accuracy                           0.59     84159
#        macro avg       0.60      0.68      0.54     84159
#     weighted avg       0.83      0.59      0.64     84159
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=31.8288, False Alarm=10.3754, Miss=21.4534
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_restaurant_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.7705
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.15      0.94      0.26     10640
#                1       0.98      0.32      0.49     81933
    
#         accuracy                           0.39     92573
#        macro avg       0.57      0.63      0.37     92573
#     weighted avg       0.88      0.39      0.46     92573
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=46.2193, False Alarm=2.6374, Miss=43.5819
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_socio_field_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.8658
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.48      0.78      0.60     21498
#                1       0.93      0.78      0.85     80538
    
#         accuracy                           0.78    102036
#        macro avg       0.71      0.78      0.72    102036
#     weighted avg       0.84      0.78      0.79    102036
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=22.2196, False Alarm=15.4213, Miss=6.7983
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_socio_lab_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.8687
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.56      0.72      0.63     20400
#                1       0.91      0.84      0.87     71016
    
#         accuracy                           0.81     91416
#        macro avg       0.74      0.78      0.75     91416
#     weighted avg       0.84      0.81      0.82     91416
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=23.5663, False Alarm=19.5149, Miss=4.0514
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:116] ================= dh3_eval_webvideo_manifest =================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:117] AUROC: 0.7967
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.45      0.76      0.57     24074
#                1       0.89      0.68      0.77     69263
    
#         accuracy                           0.70     93337
#        macro avg       0.67      0.72      0.67     93337
#     weighted avg       0.78      0.70      0.72     93337
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:125] Detection Error Rate: DetER=31.1006, False Alarm=18.1503, Miss=12.9503
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:126] ==========================================
    
    
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:128] ================== Aggregrated Results ===================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:132] ============================================================
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:133]  DetER=31.1006, False Alarm=18.1503, Miss=12.9503
# [NeMo I 2023-02-20 15:31:03 infer_seg_vad:134] ============================================================
# [NeMo I 2023-02-20 15:31:06 infer_seg_vad:139] AUROC: 0.7855
# [NeMo I 2023-02-20 15:31:06 infer_seg_vad:140] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:31:06 infer_seg_vad:141] 
#                   precision    recall  f1-score   support
    
#                0       0.50      0.59      0.54    288887
#                1       0.90      0.86      0.88   1196470
    
#         accuracy                           0.81   1485357
#        macro avg       0.70      0.73      0.71   1485357
#     weighted avg       0.82      0.81      0.81   1485357
    
# [NeMo I 2023-02-20 15:31:06 infer_seg_vad:143] {'onset': 0.5, 'offset': 0.2, 'pad_onset': 0.3, 'pad_offset': 0.3, 'min_duration_on': 0.1, 'min_duration_off': 0.1, 'filter_speech_first': True}
# [NeMo I 2023-02-20 15:31:06 infer_seg_vad:144] Done.
# /media/data/projects/NeMo-fvad/vad_code/vad_multilingual_marblenet
# (develop) ➜  vad_code git:(frame_vad) ✗ 



# [NeMo I 2023-02-20 15:45:20 infer_seg_vad:101] Finalizing individual results...
# [NeMo I 2023-02-20 15:45:24 infer_seg_vad:116] ================= voxconv_test_full_manifest =================
# [NeMo I 2023-02-20 15:45:24 infer_seg_vad:117] AUROC: 0.9329
# [NeMo I 2023-02-20 15:45:24 infer_seg_vad:118] Classification report with threshold=0.50
# [NeMo I 2023-02-20 15:45:24 infer_seg_vad:119]               precision    recall  f1-score   support
    
#                0       0.68      0.71      0.69    196226
#                1       0.97      0.96      0.96   1762905
    
#         accuracy                           0.94   1959131
#        macro avg       0.82      0.83      0.83   1959131
#     weighted avg       0.94      0.94      0.94   1959131
    
# [NeMo I 2023-02-20 15:45:24 infer_seg_vad:125] Detection Error Rate: DetER=7.2824, False Alarm=6.1515, Miss=1.1309
