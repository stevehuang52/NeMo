DATA_DIR="/home/heh/datasets/slurp_draco"
CURR_DIR="/home/heh/github/NeMo/codes"
CUDA_VISIBLE_DEVICES=1 python run_slu_to_asr_bpe.py \
    --config-path="./configs" --config-name=conformer_transformer_bpe2 \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_real_slu2asr.json,${DATA_DIR}/train_synth_slu2asr.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/dev_slu2asr.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_slu2asr.json" \
    model.tokenizer.dir="${DATA_DIR}/tokenizers_slu2asr/tokenizer_spe_unigram_v58_pad_bos_eos" \
    trainer.devices=1 \
    trainer.max_epochs=50 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=false
    # model.ssl_pretrained.model="${CURR_DIR}/nemo_experiments/ConformerL_ctc_slurp/drc_SLURP_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep600_asr_dgx1/drc_SLURP_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep600_asr_dgx1-averaged.nemo" \
    # name="slurp_conformer_large_transformer_CosineAnneal_lr3e-4x1e-4_dec3_d2048h8"
