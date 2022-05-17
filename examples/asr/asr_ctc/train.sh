python speech_to_text_ctc.py \
    --config-path=../conf --config-name=config \
    data_prefix="/home/heh/github/NeMo/tutorials/asr" \
    model.train_ds.manifest_filepath=\'./an4/train_manifest.json,./an4/train_manifest.json\' \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.devices=2 \
    trainer.max_epochs=2 \
    