CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_debug_train.py \
    mode=torchrun
