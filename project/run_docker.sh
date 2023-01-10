echo $PWD
docker run --gpus all -it --rm --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --ipc=host \
-v $PWD:/NeMo \
-v $PWD/project/synth_audio_train:$PWD/project/synth_audio_train \
-v $PWD/project/synth_audio_val:$PWD/project/synth_audio_val \
-v /media/data/datasets/vad_sd:/media/data/datasets/vad_sd \
gitlab-master.nvidia.com/heh/nemo_containers:nemo-1.13-22.11 /bin/bash

# gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-22.11-nb531 /bin/bash
# gitlab-master.nvidia.com/hainanx/nemo_containers:main
# gitlab-master.nvidia.com/heh/nemo_containers:nemo-1.13-22.11 /bin/bash 
# gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-22.10-notrt /bin/bash
# nvcr.io/nvidia/pytorch:22.08-py3
# gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-22.09 /bin/bash
# cmd="cd /NeMo/project && PL_DISABLE_FORK=1 python run_debug_train.py"
