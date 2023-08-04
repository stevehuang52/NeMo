NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo

TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json

python run_sft_audio_lm.py --config-path="./configs" --config-name "audio_gpt_debug" \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.file_names=$TRAIN_MANIFESTS \
    model.data.validation_ds.file_names=$VAL_MANIFESTS

