set -x
# Container to be used by job. TIP: create "complete" containers, do not do pip/apt install as part of job, 22sep12
CONTAINER="gitlab-master.nvidia.com/heh/nemo_containers:nemo-22sep12"

#### Project Constants ####
PROJECT_NAME=Stream_VAD
DATASET=Multilang
###########################


#### Model Initialization ####
INIT_EXP=''
INIT_MODEL=''
##############################

#### Hyper-Parameters ##########
GRAD_ACC=1
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128
MAX_EPOCHS=50
SAVE_TOP_K=5
LOG_PREDICTION=true
PRECISION=32
OPT=sgd
LR=1e-2
WD=1e-3
GRAD_CLIP=0.0
# Misc
MODEL_POSFIX=''
###############################

##### Dataset ####
# Paths are after mounted
DATA_ROOT="/manifests"
TRAIN_WORKERS=8
TRAIN_ISTARRED=false
TRAIN_MANIFEST="${DATA_ROOT}/mandarin_40ms_train.json"
TRAIN_FILEPATHS='na'

VAL_WORKERS=8
VAL_ISTARRED=false
VAL_MANIFEST="${DATA_ROOT}/mandarin_40ms_dev.json"
VAL_FILEPATHS='na'

TEST_WORKERS=8
TEST_ISTARRED=false
TEST_MANIFEST="${DATA_ROOT}/mandarin_40ms_dev.json"
TEST_FILEPATHS='na'
##################

##### Code&Config Location ####
CODE_DIR=/media/data/projects/NeMo-vad
CONFIG_PATH='./configs/'
CONFIG_NAME=marblenet_3x2x64
###############################



################ Usually No Change #########################################################################
NOW=$(date +'%m/%d/%Y-%T')
### Experiment Name ###
EXP_NAME=docker_${DATASET}_${OPT}lr${LR}_wd${WD}_aug${TIME_MASKS}x${TIME_WIDTH}_b${TRAIN_BATCH_SIZE}_gacc${GRAD_ACC}_ep${MAX_EPOCHS}${MODEL_POSFIX}

######## Local Paths Before Mapping ###################

# Where training/validation data is, various manifests and configs
DATA_DIR=/media/data/datasets

# personal manifests
MANIFESTS_DIR=/media/data/projects/NeMo-vad/project/manifests_draco

# Where to store results and logs
RESULTS_DIR=/media/data/projects/NeMo-vad/project/nemo_experiments/${EXP_NAME}
#########################################################

mkdir -p $RESULTS_DIR
OUTFILE=${RESULTS_DIR}/output-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# actually mount folders to container
MOUNTS="--mount=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${DATA_DIR}:/data/multilingual_vad,${TOKENIZERS_DIR}:/tokenizers,${MANIFESTS_DIR}:/manifests"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& cd /code/ \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -c 'import pytorch_lightning as ptl; print(ptl.__version__)' \
&& export TOKENIZERS_PARALLELISM=false \
&& python ${SCRIPT_FILE}  \
--config-path=$CONFIG_PATH \
--config-name=$CONFIG_NAME \
exp_manager.create_wandb_logger=true \
exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME \
exp_manager.wandb_logger_kwargs.name=$EXP_NAME \
exp_manager.name=$EXP_NAME \
++exp_manager.resume_if_exists=true \
++exp_manager.resume_ignore_no_checkpoint=true \
++exp_manager.use_datetime_version=false \
++trainer.log_every_n_steps=100 \
++exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K \
++trainer.benchmark=false \
++trainer.precision=$PRECISION \
++model.log_prediction=$LOG_PREDICTION \
model.train_ds.pin_memory=true \
model.test_ds.pin_memory=true \
model.validation_ds.pin_memory=true \
model.train_ds.num_workers=$TRAIN_WORKERS \
model.validation_ds.num_workers=$VAL_WORKERS \
model.test_ds.num_workers=$TEST_WORKERS \
exp_manager.exp_dir=/results/ \
++trainer.check_val_every_n_epoch=1 \
trainer.num_nodes=$SLURM_JOB_NUM_NODES \
model.train_ds.is_tarred=$TRAIN_ISTARRED \
model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
model.validation_ds.manifest_filepath=$VAL_MANIFEST \
model.test_ds.manifest_filepath=$TEST_MANIFEST \
trainer.max_epochs=$MAX_EPOCHS \
trainer.accumulate_grad_batches=${GRAD_ACC} \
model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
model.train_ds.bucketing_batch_size=$BUCKET_BATCH_SIZE \
model.train_ds.bucketing_strategy=$BUCKET_STRATEGY \
model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
model.test_ds.batch_size=$EVAL_BATCH_SIZE \
model.optim.name=$OPT \
model.optim.lr=$LR \
model.optim.weight_decay=$WD
EOF


sudo docker run $CONTAINER \
  $MOUNTS \
  bash -c "${cmd}"
set +
