CUDA_LAUNCH_BLOCKING=1 python run_debug_train.py trainer.devices=1 mode=ptl model.train_ds.num_workers=0 model.validation_ds.num_workers=0

# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     run_debug_train.py \
#     data.skip=true mode=torch_ddp