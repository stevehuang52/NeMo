CUDA_LAUNCH_BLOCKING=1 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_debug_train.py \
    mode=torch_ddp
