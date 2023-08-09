NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

NUM_WORKERS=0
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean.json,/media/data/datasets/LibriSpeech/dev_clean.json]"

python run_sft_audio_gpt_lora.py --config-path="./configs" --config-name "megatron_audio_gpt_lora_sft" \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.validation_ds.num_workers=$NUM_WORKERS

