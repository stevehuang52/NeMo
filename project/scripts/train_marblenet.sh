DATA_DIR="/media/data/datasets/mandarin/aishell2"
NAME="marblenet_3x2x64_mandarin_40ms_092922_debug"
BATCH_SIZE=1024
CUDA_VISIBLE_DEVICES=0,1 python speech_to_multi_label.py \
    --config-path="./configs" --config-name="marblenet_3x2x64" \
    model.train_ds.manifest_filepath="${DATA_DIR}/manifests_abs/svad_mandarin_40ms_train.json" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/manifests_abs/svad_mandarin_40ms_dev.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/manifests_abs/svad_mandarin_40ms_dev.json" \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.test_ds.batch_size=$BATCH_SIZE \
    trainer.max_epochs=50 \
    exp_manager.name=${NAME} \
    exp_manager.create_wandb_logger=false \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project="Stream_VAD"
    # model.train_ds.augmentor.noise.manifest_path="/media/data/projects/NeMo-vad/project/manifests_noise/freesound_nonspeech_test_local.json"
