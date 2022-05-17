python speech_to_text_ctc_bpe.py \
    --config-path=./configs --config-name=conformer_ctc_bpe \
    model.tokenizer.dir="/home/heh/datasets/Catalan/catalan_data/tokenizers/tokenizer_spe_unigram_v128" \
    model.train_ds.batch_size=64 \
    model.train_ds.data_prefix="/home/heh/datasets/Catalan/catalan_data/tarred/s512_b1" \
    model.train_ds.manifest_filepath="tarred_audio_manifest.json" \
    model.train_ds.is_tarred=true \
    model.train_ds.tarred_audio_filepaths="/home/heh/datasets/Catalan/catalan_data/tarred/s512_b1/audio__OP_0..511_CL_.tar" \
    model.validation_ds.data_prefix="/home/heh/datasets/Catalan/catalan_data" \
    model.validation_ds.manifest_filepath="dev.json" \
    trainer.devices=1 \
    trainer.max_epochs=1000 \
    exp_manager.create_wandb_logger=false
