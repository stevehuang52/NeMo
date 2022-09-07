DATA_DIR="/home/heh/datasets/slurp_draco"
CUDA_VISIBLE_DEVICES=1 python run_slu_to_nlu.py \
    --config-path="./configs" --config-name=transformer_bpe_nlu_oracle3 \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_nlu_oracle.json,${DATA_DIR}/train_synth_nlu_oracle.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_nlu_oracle.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_nlu_oracle.json" \
    model.asr_tokenizer.dir="${DATA_DIR}/tokenizers_nlu/tokenizer_spe_unigram_v1024_pad_bos_eos" \
    model.nlu_tokenizer.dir="${DATA_DIR}/tokenizers_slu2asr/tokenizer_spe_unigram_v512_pad_bos_eos" \
    trainer.devices=1 \
    trainer.max_epochs=30 \
    model.optim.sched.warmup_steps=500 \
    exp_manager.create_wandb_logger=true
