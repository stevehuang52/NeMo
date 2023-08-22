NEMO_DIR=/home/heh/codes/nemo-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo

# must have a `\` before `=` to avoid hydra errors, the eval script will remove the `\`
# ALM_CKPT='/home/heh/codes/nemo-slm/workspace/nemo_experiments/AudioGPT-LS-nontar-dev-clean-debug-1sample-r4/checkpoints/AudioGPT-LS-nontar-dev-clean-debug-1sample-r4--validation_wer\=0.000-step=164.ckpt'
# ALM_YAML='/home/heh/codes/nemo-slm/workspace/nemo_experiments/AudioGPT-LS-nontar-dev-clean-debug-1sample-r4/version_3/hparams.yaml'

ALM_CKPT="'/home/heh/codes/nemo-slm/workspace/nemo_experiments/AudioGPT-LS-nontar-nospg-debug-full-r2/checkpoints/AudioGPT-LS-nontar-nospg-debug-full-r2--validation_wer=1.299-step\=259148-last.ckpt'"
ALM_YAML='/home/heh/codes/nemo-slm/workspace/nemo_experiments/AudioGPT-LS-nontar-nospg-debug-full-r2/version_0/hparams.yaml'

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json,/media/data/datasets/LibriSpeech/debug_1.json]"
# VAL_NAMES=[debug-1,debug-1]

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_cleaned.json,/media/data/datasets/LibriSpeech/dev_other.json,/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json]"
# VAL_NAMES="[dev-clean,dev-other,train-clean-100]"

VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json]"
VAL_NAMES="[train-clean-100]"

HYDRA_FULL_ERROR=1 python eval_audio_gpt_lora.py \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.test_ds.names=$VAL_NAMES \
    model.data.test_ds.global_batch_size=32 \
	model.data.test_ds.micro_batch_size=32 \
	model.data.test_ds.tokens_to_generate=128
