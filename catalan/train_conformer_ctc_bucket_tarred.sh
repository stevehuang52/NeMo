DATA_ROOT="/home/heh/datasets/Catalan/catalan_data/tarred/s512_b4/"
python speech_to_text_ctc_bpe.py \
    --config-path=./configs --config-name=conformer_ctc_bpe \
    model.tokenizer.dir="/home/heh/datasets/Catalan/catalan_data/tokenizers/tokenizer_spe_unigram_v128" \
    model.train_ds.is_tarred=true \
    model.train_ds.batch_size=1 \
    model.train_ds.bucketing_batch_size=[40,32,24,16] \
    model.train_ds.manifest_filepath=[["${DATA_ROOT}/bucket1/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket2/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket3/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket4/tarred_audio_manifest.json"]] \
    model.train_ds.tarred_audio_filepaths=[["${DATA_ROOT}/bucket1/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket2/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket3/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket4/audio__OP_0..511_CL_.tar"]] \
    model.validation_ds.data_prefix="/home/heh/datasets/Catalan/catalan_data" \
    model.validation_ds.manifest_filepath="dev.json" \
    trainer.devices=1 \
    trainer.max_epochs=1000 \
    exp_manager.create_wandb_logger=false
