

data_dir=./manifests_test
out_dir=/media/data/projects/NeMo-fvad/vad_code/nemo_experiments
CUDA_VISIBLE_DEVICES=1 python infer_seg_vad.py \
    --config-path="./configs" --config-name="vad_seg_inference" \
    frame_out_dir="${out_dir}/segment_vad_multi_output" \
    dataset="[${data_dir}/ava_eval_10ms.json,${data_dir}/ami_eval_10ms.json,${data_dir}/ch120_CH109_10ms.json]"

    # dataset="[${data_dir}/french_test_20ms.json,${data_dir}/german_test_20ms.json,${data_dir}/mandarin_test_20ms.json,${data_dir}/russian_test_20ms.json,${data_dir}/spanish_test_20ms.json]"