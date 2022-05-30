DATA_ROOT="/home/heh/datasets/Catalan/catalan_data/"
python speech_to_text_ctc_bpe.py \
    --config-path=./configs --config-name=conformer_ctc_bpe \
    model.tokenizer.dir="/home/heh/datasets/Catalan/catalan_data/tokenizers/tokenizer_spe_unigram_v128" \
    model.train_ds.batch_size=32 \
    model.train_ds.manifest_filepath="${DATA_ROOT}/train.json" \
    model.validation_ds.manifest_filepath="${DATA_ROOT}/dev.json" \
    trainer.devices=2 \
    trainer.max_epochs=1000 \
    exp_manager.create_wandb_logger=false
