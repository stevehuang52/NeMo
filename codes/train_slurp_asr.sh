DATA_DIR="/home/heh/datasets/slurp_draco"
CUDA_VISIBLE_DEVICES=1 python ../examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path="../conf/conformer" --config-name=conformer_ctc_bpe \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_asr.json,${DATA_DIR}/train_synth_asr.json]" \
    model.train_ds.batch_size=32 \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_asr.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_asr.json" \
    model.tokenizer.dir="${DATA_DIR}/tokenizers_asr/tokenizer_spe_unigram_v128" \
    model.optim.weight_decay=1e-3 \
    trainer.devices=1 \
    trainer.max_epochs=300 \
    model.optim.sched.warmup_steps=5000 \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name="stt_conformer_ctc_large_slurp" \
    exp_manager.wandb_logger_kwargs.project="ASR_SLURP" \
    name="stt_conformer_ctc_large_slurp"
