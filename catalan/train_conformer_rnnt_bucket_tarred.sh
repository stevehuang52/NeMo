DATA_ROOT="/home/heh/datasets/Catalan/catalan_data/tarred/s512_b4/"
CUDA_VISIBLE_DEVICES=0,1 python speech_to_text_rnnt_bpe.py \
    --config-path=./configs --config-name=conformer_transducer_bpe \
    model.tokenizer.dir="/home/heh/datasets/Catalan/catalan_data/tokenizers/tokenizer_spe_unigram_v1024" \
    model.train_ds.is_tarred=true \
    model.train_ds.batch_size=1 \
    model.train_ds.num_workers=4 \
    model.train_ds.bucketing_batch_size=[64,48,32,16] \
    model.train_ds.bucketing_strategy=fully_randomized \
    model.train_ds.manifest_filepath=[["${DATA_ROOT}/bucket1/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket2/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket3/tarred_audio_manifest.json"],\
["${DATA_ROOT}/bucket4/tarred_audio_manifest.json"]] \
    model.train_ds.tarred_audio_filepaths=[["${DATA_ROOT}/bucket1/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket2/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket3/audio__OP_0..511_CL_.tar"],\
["${DATA_ROOT}/bucket4/audio__OP_0..511_CL_.tar"]] \
    model.validation_ds.manifest_filepath="/home/heh/datasets/Catalan/catalan_cleaned/dev/dev.json" \
    trainer.devices=2 \
    trainer.max_epochs=1 \
    trainer.log_every_n_steps=1 \
    model.log_prediction=false \
    model.joint.fuse_loss_wer=true \
    model.joint.fused_batch_size=8 \
    exp_manager.create_wandb_logger=false
