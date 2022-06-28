DATA_DIR="/home/heh/datasets/slurp-speechbrain"
CUDA_VISIBLE_DEVICES=0,1 python run_slu_to_asr_bpe.py \
    --config-path="." --config-name=conformer_transformer_bpe \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_slu2asr.json,${DATA_DIR}/train_synth_slu2asr.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_slu2asr.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_slu2asr.json" \
    model.tokenizer.dir="/home/heh/datasets/slurp-speechbrain/tokenizers_s2s/tokenizer_spe_unigram_v58_pad_bos_eos" \
    trainer.devices=2 \
    trainer.max_epochs=200 \
    model.optim.sched.warmup_steps=1000 \
    exp_manager.create_wandb_logger=True
