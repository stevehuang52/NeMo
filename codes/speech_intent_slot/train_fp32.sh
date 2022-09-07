DATA_DIR="/home/heh/datasets/slurp_draco"
# DATA_DIR="/home/heh/datasets/slurp_data"
EXP_NAME="slurp_conformer_transformer_large_fp32"
CUDA_VISIBLE_DEVICES=0,1 python run_speech_intent_slot_train.py \
    --config-path="./configs" --config-name=conformer_transformer_large_bpe \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_slu2asr.json,${DATA_DIR}/train_synth_slu2asr.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_slu2asr.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_slu2asr.json" \
    model.tokenizer.dir="${DATA_DIR}/tokenizers_slu2asr/tokenizer_spe_unigram_v58_pad_bos_eos" \
    model.train_ds.batch_size=32 \
    trainer.devices=1 \
    trainer.max_epochs=50 \
    exp_manager.name=$EXP_NAME \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.create_wandb_logger=false \
    exp_manager.wandb_logger_kwargs.name=$EXP_NAME \
    exp_manager.wandb_logger_kwargs.project="SLURP_SLU2ASR"
