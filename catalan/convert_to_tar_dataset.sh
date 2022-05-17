export CAT_DATA_ROOT="/home/heh/datasets/Catalan/catalan_data"
python ./convert_to_tarred_audio_dataset.py \
    --manifest_path "${CAT_DATA_ROOT}/train.json" \
    --target_dir "${CAT_DATA_ROOT}/tarred/s512_b4"\
    --num_shards 512 \
    --max_duration 11.0 \
    --min_duration 1.0 \
    --workers 24 \
    --buckets_num 4