NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

EXP_NAME=AudioGPT-tarred-LS-debug2

GLOBAL_BATCH=256
MICRO_BATCH=64

NUM_WORKERS=2
TRAIN_MANIFESTS="/media/data3/librispeech_tarred/tarred_audio_manifest.json"
TRAIN_FILEPATHS="/media/data3/librispeech_tarred/audio__OP_0..511_CL_.tar"
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean.json,/media/data/datasets/LibriSpeech/dev_clean.json]"

python run_sft_audio_gpt_lora.py --config-path="./configs" --config-name "megatron_audio_gpt_lora_sft" \
    name=$EXP_NAME \
    trainer.devices=-1 \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.is_tarred=True \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.tarred_audio_filepaths=${TRAIN_FILEPATHS} \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.validation_ds.num_workers=$NUM_WORKERS

