CUDA_LAUNCH_BLOCKING=1 python run_debug_train.py trainer.devices=1 mode=ptl

# RUN with pytorch DDP, w/o PTL
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     run_debug_train.py \
#     data.skip=true mode=torch_ddp