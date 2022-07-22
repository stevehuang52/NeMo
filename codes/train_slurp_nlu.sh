DATA_DIR="/home/heh/datasets/slurp_draco"
CUDA_VISIBLE_DEVICES=1 python run_slu_to_nlu.py \
    --config-path="./configs" --config-name=conformer_transformer_bpe_nlu \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_nlu.json,${DATA_DIR}/train_synth_nlu.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_nlu.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_nlu.json" \
    model.tokenizer.dir="/home/heh/datasets/slurp-speechbrain/tokenizers_s2s/tokenizer_spe_unigram_v58_pad_bos_eos" \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=True
