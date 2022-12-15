echo $PWD
docker run --gpus all -it --rm --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --ipc=host \
-v $PWD:/code \
-v $PWD/synth_audio_train:$PWD/synth_audio_train \
-v $PWD/synth_audio_val:$PWD/synth_audio_val \
nemo-main-debug /bin/bash
