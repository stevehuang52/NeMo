# Debugging Error in Training with PTL


## Bug Description

We encountered "zipfile.BadZipFile: Truncated file header" error while training NeMo models with PytorchLightning. The bug disappeared after downgrading numba from 0.56.2 to 0.53.1. Detailed error message is in `bug-nemo1_13-ptl1_8-swdl22_11.txt`.


## Steps to Reproduce the Error
Under the NeMo git root directory:

1. Build a docker image with the Dockerfile: `DOCKER_BUILDKIT=1 docker build . -f Dockerfile -t nemo-main-debug`
2. Start the docker container 
```
docker run --gpus all -it --rm --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --ipc=host \
    -v $PWD:/code \
    -v $PWD/synth_audio_train:$PWD/synth_audio_train \
    -v $PWD/synth_audio_val:$PWD/synth_audio_val \
    nemo-main-debug /bin/bash
```
3. Start the debug script: `cd nemo_debug && ./run_debug.sh`


