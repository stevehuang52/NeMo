DATA_DIR="/home/heh/datasets/slurp_draco"
CUDA_VISIBLE_DEVICES=0 python run_slu_to_nlu.py \
    --config-path="./configs" --config-name=conformer_transformer_bpe_nlu_oracle2 \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_nlu_oracle.json,${DATA_DIR}/train_synth_nlu_oracle.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_nlu_oracle.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_nlu_oracle.json" \
    model.tokenizer.dir="/home/heh/datasets/slurp-speechbrain/tokenizers_s2s/tokenizer_spe_unigram_v58_pad_bos_eos" \
    trainer.devices=1 \
    trainer.max_epochs=50 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=true
