python speech_to_text_ctc.py \
    --config-path=../conf/conformer --config-name=conformer_ctc_char \
    data_prefix="/home/heh/datasets/LibriSpeech" \
    model.train_ds.manifest_filepath="train_clean_100.json" \
    model.validation_ds.manifest_filepath="dev_clean.json" \
    trainer.devices=2 \
    trainer.max_epochs=50 \
    