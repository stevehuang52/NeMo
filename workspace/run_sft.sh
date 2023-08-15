NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

GLOBAL_BATCH=1
MICRO_BATCH=1
NUM_WORKERS=0
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json,/media/data/datasets/LibriSpeech/train_clean_360_cleaned.json,/media/data/datasets/LibriSpeech/train_other_500_cleaned.json]
# TRAIN_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean.json]"
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_other.json]"
# TRAIN_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json]"
# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json]"
EXP_NAME=AudioGPT-LS-nontar-dev-clean-debug-r5
PROJECT_NAME=audio-text-llm-debug

# wandb login f5029311df02a27459c2c99c5fbef08978dc709e

CUDA_VISIBLE_DEVICES=1 python run_sft_audio_gpt_lora.py --config-path="./configs" --config-name "megatron_audio_gpt_lora_sft" \
    name=$EXP_NAME \
    ++exp_manager.create_wandb_logger=false \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=false \
    trainer.devices=1 \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS

