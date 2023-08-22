NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

GLOBAL_BATCH=64
MICRO_BATCH=32
NUM_WORKERS=0
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json,/media/data/datasets/LibriSpeech/train_clean_360_cleaned.json,/media/data/datasets/LibriSpeech/train_other_500_cleaned.json]
# TRAIN_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean.json]"
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_cleaned.json,/media/data/datasets/LibriSpeech/dev_other.json]"
VAL_NAMES="[dev-clean,dev-other]"
# TRAIN_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json]"
# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json]"
EXP_NAME=AudioGPT-LS-nontar-debug-full-r3
PROJECT_NAME=audio-text-llm-debug


CUDA_VISIBLE_DEVICES='0,1' python run_sft_audio_gpt_lora.py --config-path="./configs" --config-name "megatron_audio_gpt_lora_sft" \
    name=$EXP_NAME \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=false \
    trainer.devices=-1 \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.validation_ds.names=$VAL_NAMES

