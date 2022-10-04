

## TODO
- [x] Visualize multilingual test data predictions & labels to see where the model falls short

- [x] Prepare and cut AMI manifest for train/dev/test
- [x] Prepare and cut isci for training

- [x] Prepare and cut fisher 2004 manifest for train
- [x] Prepare and cut fisher 2005 manifest for train

- [x] Prepare and cut ch120-Ch109 manifest for test
- [x] Prepare and cut ch120-moved for dev


## Changed Files
- nemo.collections.asr.data.audio_to_label
- nemo.collections.asr.data.audio_to_label_dataset




```shell
python tools/clean_manifest.py manifests_draco/french_train_40ms.json \

```


## Data Processing
1. Generate manifest with: audio_filepath, duration, offset
2. For each manifest item, load RTTM and obtain frame-level labels





[NeMo I 2022-10-04 14:56:10 infer_vad_multi:116] ================= french_test_20ms =================
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:117] AUROC: 0.9558
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.83      0.50      0.63    568456
               1       0.91      0.98      0.94   2904042

        accuracy                           0.90   3472498
       macro avg       0.87      0.74      0.79   3472498
    weighted avg       0.90      0.90      0.89   3472498

[NeMo I 2022-10-04 14:56:10 infer_vad_multi:125] Detection Error Rate: DetER=11.8504, False Alarm=10.2319, Miss=1.6185
[NeMo I 2022-10-04 14:56:10 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:116] ================= german_test_20ms =================
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:117] AUROC: 0.9634
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.85      0.73      0.78   2893332
               1       0.89      0.94      0.92   6700110

        accuracy                           0.88   9593442
       macro avg       0.87      0.84      0.85   9593442
    weighted avg       0.88      0.88      0.88   9593442

[NeMo I 2022-10-04 14:56:32 infer_vad_multi:125] Detection Error Rate: DetER=16.0786, False Alarm=9.7779, Miss=6.3007
[NeMo I 2022-10-04 14:56:32 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:116] ================= mandarin_test_20ms =================
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:117] AUROC: 0.8878
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.84      0.43      0.57    365268
               1       0.84      0.97      0.90   1081786

        accuracy                           0.84   1447054
       macro avg       0.84      0.70      0.74   1447054
    weighted avg       0.84      0.84      0.82   1447054

[NeMo I 2022-10-04 14:56:35 infer_vad_multi:125] Detection Error Rate: DetER=21.3148, False Alarm=18.6094, Miss=2.7054
[NeMo I 2022-10-04 14:56:35 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:116] ================= russian_test_20ms =================
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:117] AUROC: 0.9605
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.86      0.63      0.73   1389584
               1       0.89      0.97      0.93   4339820

        accuracy                           0.88   5729404
       macro avg       0.87      0.80      0.83   5729404
    weighted avg       0.88      0.88      0.88   5729404

[NeMo I 2022-10-04 14:56:46 infer_vad_multi:125] Detection Error Rate: DetER=14.1586, False Alarm=10.5631, Miss=3.5955
[NeMo I 2022-10-04 14:56:46 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:116] ================= spanish_test_20ms =================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:117] AUROC: 0.9533
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:118] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:119]               precision    recall  f1-score   support

               0       0.83      0.49      0.62    526170
               1       0.92      0.98      0.95   3078748

        accuracy                           0.91   3604918
       macro avg       0.87      0.74      0.78   3604918
    weighted avg       0.91      0.91      0.90   3604918

[NeMo I 2022-10-04 14:56:53 infer_vad_multi:125] Detection Error Rate: DetER=10.4260, False Alarm=9.0447, Miss=1.3813
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:126] ==========================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:128] ================== Aggregrated Results ===================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:132] ============================================================
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:133]  DetER=10.4260, False Alarm=9.0447, Miss=1.3813
[NeMo I 2022-10-04 14:56:53 infer_vad_multi:134] ============================================================
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:139] AUROC: 0.9573
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:140] Classification report with threshold=0.50
[NeMo I 2022-10-04 14:57:48 infer_vad_multi:141]
                  precision    recall  f1-score   support

               0       0.85      0.64      0.73   5742810
               1       0.89      0.96      0.93  18104506

        accuracy                           0.89  23847316
       macro avg       0.87      0.80      0.83  23847316
    weighted avg       0.88      0.89      0.88  23847316
