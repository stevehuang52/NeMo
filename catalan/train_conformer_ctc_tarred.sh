python speech_to_text_ctc.py \
    --config-path=./configs --config-name=conformer_ctc_char \
    model.train_ds.batch_size=32 \
    model.train_ds.is_tarred=true \
    model.train_ds.manifest_filepath="/home/heh/datasets/Catalan/catalan_cleaned/train_tarred/s512_b1/tarred_audio_manifest.json" \
    model.train_ds.tarred_audio_filepaths="/home/heh/datasets/Catalan/catalan_cleaned/train_tarred/s512_b1/audio__OP_0..511_CL_.tar" \
    model.validation_ds.manifest_filepath="/home/heh/datasets/Catalan/catalan_cleaned/dev/dev.json" \
    trainer.devices=2 \
    trainer.max_epochs=1000 \
    exp_manager.create_wandb_logger=false

#    model.tokenizer.dir="/home/heh/datasets/Catalan/catalan_data/tokenizers/tokenizer_spe_unigram_v128" \
