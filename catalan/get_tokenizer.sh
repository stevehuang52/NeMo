export CAT_DATA_ROOT="/home/heh/datasets/Catalan/catalan_data"
python ./process_asr_text_tokenizer.py \
  --manifest="${CAT_DATA_ROOT}/train_nopunc.json" \
  --data_root="${CAT_DATA_ROOT}/tokenizers/" \
  --vocab_size=1024 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log




sudo DOCKER_BUILDKIT=1 docker build --no-cache . -f Dockerfile -t nemo-22may16

sudo docker image tag nemo-22may16 gitlab-master.nvidia.com/heh/nemo_containers:nemo-22may16

rsync -Wav manifest/ heh@draco1:/gpfs/fs1/projects/ent_aiapps/datasets/data/ASR/catalan/cv-2022-04-27/manifest

rsync -Wav tokenizers/ heh@draco1:/gpfs/fs1/projects/ent_aiapps/datasets/tokenizers/ASR/catalan/cv-2022-04-27


gitlab-master.nvidia.com/heh/nemo_containers:nemo-dev

sudo docker push gitlab-master.nvidia.com/heh/nemo_containers:nemo-22may16

sudo docker image tag nemo-dev gitlab-master.nvidia.com/heh/nemo_containers:nemo-dev