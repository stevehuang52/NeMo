sudo docker run --gpus all -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd -it \
-v /media/data/projects/NeMo-vad:/NeMo \
-v /media/data/datasets/vad_sd:/media/data/datasets/vad_sd \
nemo-vad /bin/bash
