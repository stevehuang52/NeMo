#!/bin/bash
#SBATCH -A ent_aiapps_asr
#SBATCH -p batch_dgx1_m2       # batch / batch_short / backfill, for mod3 use batch_dgx1_m3, but m3 has both 16GB and 32GB GPUS
#SBATCH -N 1                    # number of nodes (32 nodes max for draco)
#SBATCH -t 8:00:00              # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --gpus-per-node=8      # n gpus per machine <required>
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --nv-meta=ml-model.CONFORMERCTC

set -x

####### Common Constants ######
SLURM_ACCOUNT='ent_aiapps'
USERID='users/heh'
WANDB_TOKEN="f5029311df02a27459c2c99c5fbef08978dc709e"
LUSTRE_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}  # /lustre/fs1 for mod3
CLUSTER='drc'
###############################

# Container to be used by job. TIP: create "complete" containers, do not do pip/apt install as part of job
CONTAINER="gitlab-master.nvidia.com/heh/nemo_containers:nemo-22may16"

#### Project Constants ####
PROJECT_NAME="SLURP_SLU2ASR"
DATASET="SLURP"
MODEL_NAME="ConformerL-Transformer"
###########################


#### Model Initialization ####
INIT_EXP=''
INIT_MODEL=''
##############################

#### Hyper-Parameters ##########
MAX_EPOCHS=100
GRAD_ACC=1
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
BUCKET_BATCH_SIZE=[40,32,24,16]
BUCKET_STRATEGY=fully_randomized  # synced_randomized, fully_randomized
TIME_MASKS=10
TIME_WIDTH=0.05

# Optimization
OPT=adamw
LR=0.0003
LR_ENC=1e-4
WD=0.0
SCHEDULER="CosineAnnealing"  # WarmupAnnealing, CosineAnnealing
WARMUP=2000
GRAD_CLIP=0.0

# Model
FREEZE_ENCODER=false
DECODER_LAYERS=3
DECODER_INNER_SIZE=2048
DECODER_FFT_DROPOUT=0.1
DECODER_ATTN_SCORE_DROPOUT=0.0
DECODER_ATTN_LAYER_DROPOUT=0.0

# Misc
SAVE_TOP_K=5
LOG_PREDICTION=true
SUBSAMPLING=striding # stacking
PRECISION=32
MODEL_POSFIX=''
###############################

##### Dataset ####
DATA_ROOT="/data/SLU/slurp"
# Paths are after mounted
TOKENIZER="${DATA_ROOT}/tokenizers_slu2asr/tokenizer_spe_unigram_v58_pad_bos_eos"
MAX_DURATION=10.0

TRAIN_WORKERS=4
TRAIN_ISTARRED=false
# Non-bucketing
TRAIN_MANIFEST="[${DATA_ROOT}/train_real_slu2asr.json,${DATA_ROOT}/train_synth_slu2asr.json]"
TRAIN_FILEPATHS='na'
# TRAIN_FILEPATHS=${TRAIN_DATA_PATH}/audio__OP_0..511_CL_.tar
# Bucketing, set batch_size to 1, use BUCKET_BATCH_SIZE instead
# TRAIN_MANIFEST="[[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket1/tarred_audio_manifest.json],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket2/tarred_audio_manifest.json],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket3/tarred_audio_manifest.json],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket4/tarred_audio_manifest.json]]"
# TRAIN_FILEPATHS="[[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket1/audio__OP_0..511_CL_.tar],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket2/audio__OP_0..511_CL_.tar],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket3/audio__OP_0..511_CL_.tar],[/data/MozillaCommonVoice/Catalan/train_tarred/512shard_4bucket/bucket4/audio__OP_0..511_CL_.tar]]"

VAL_WORKERS=4
VAL_ISTARRED=false
VAL_MANIFEST="${DATA_ROOT}/dev_slu2asr.json"
VAL_FILEPATHS='na'

TEST_WORKERS=4
TEST_ISTARRED=false
TEST_MANIFEST="${DATA_ROOT}/test_slu2asr.json"
TEST_FILEPATHS='na'
##################

##### Code&Config Location ####
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/code/nemo-slu
CONFIG_PATH='./configs/'
CONFIG_NAME=conformer_transformer_bpe
###############################



################ Usually No Change #########################################################################
NOW=$(date +'%m/%d/%Y-%T')
### Experiment Name ###
EXP_NAME=${CLUSTER}_${MODEL_NAME}_dl${DECODER_LAYERS}_${OPT}lr${LR}_wd${WD}_gc${GRAD_CLIP}_${SCHEDULER}_wp${WARMUP}_aug${TIME_MASKS}x${TIME_WIDTH}_b${TRAIN_BATCH_SIZE}_ep${MAX_EPOCHS}${MODEL_POSFIX}

######## Local Paths Before Mapping ###################
INIT_EXP_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/

# Where training/validation data is, various manifests and configs
DATA_DIR=${LUSTRE_ACCOUNT_PREFIX}/datasets/data/

#TOKENIZERS_DIR=${LUSTRE_ACCOUNT_PREFIX}/users/${USERID}/tokenizers
TOKENIZERS_DIR=${LUSTRE_ACCOUNT_PREFIX}/datasets/tokenizers

# personal manifests
MANIFESTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/manifests

# Where to store results and logs
RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME
#########################################################

mkdir -p $RESULTS_DIR
OUTFILE=${RESULTS_DIR}/output-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# actually mount folders to container
MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${DATA_DIR}:/data,${TOKENIZERS_DIR}:/tokenizers,${MANIFESTS_DIR}:/manifests,${INIT_EXP_DIR}:/exp_init"

if [ -z "$INIT_MODEL" ]
then
  INIT_CHECKPOINT=''
else
  INIT_CHECKPOINT=/exp_init/$INIT_MODEL
  #CHECKPOINT_DIR=$INIT_EXP_DIR/$INIT_EXP/$INIT_EXP/checkpoints/
  #INIT_CHECKPOINT=/exp_init/$INIT_EXP/$INIT_EXP/checkpoints/$(ls -l $CHECKPOINT_DIR | grep '\-last.ckpt' | awk '{print $9}')
fi

### Script to Run ###
SCRIPT_FILE="codes/run_slu_to_asr_bpe.py"
############################################################################################################


# Your actual script. Note that paths are inside container.
# Change CUDA_VISIBLE_DEVICES based on your cluster node config
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& wandb login ${WANDB_TOKEN} \
&& cd /code/ \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:${PYTHONPATH}" \
&& python -c 'import pytorch_lightning as ptl; print(ptl.__version__)' \
&& export TOKENIZERS_PARALLELISM=false \
&& python ${SCRIPT_FILE}  \
--config-path=$CONFIG_PATH \
--config-name=$CONFIG_NAME \
exp_manager.create_wandb_logger=true \
exp_manager.wandb_logger_kwargs.project=$PROJECT_NAME \
exp_manager.wandb_logger_kwargs.name=$EXP_NAME \
exp_manager.name=$EXP_NAME \
exp_manager.resume_if_exists=true \
exp_manager.resume_ignore_no_checkpoint=true \
+exp_manager.use_datetime_version=false \
trainer.log_every_n_steps=100 \
trainer.progress_bar_refresh_rate=100 \
exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K \
++trainer.benchmark=false \
trainer.precision=$PRECISION \
model.log_prediction=$LOG_PREDICTION \
model.tokenizer.dir=$TOKENIZER \
model.train_ds.pin_memory=true \
model.test_ds.pin_memory=true \
model.validation_ds.pin_memory=true \
model.train_ds.num_workers=$TRAIN_WORKERS \
model.validation_ds.num_workers=$VAL_WORKERS \
model.test_ds.num_workers=$TEST_WORKERS \
exp_manager.exp_dir=/results/ \
trainer.check_val_every_n_epoch=1 \
trainer.num_nodes=$SLURM_JOB_NUM_NODES \
model.train_ds.max_duration=${MAX_DURATION} \
model.train_ds.is_tarred=$TRAIN_ISTARRED \
model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
model.validation_ds.manifest_filepath=$VAL_MANIFEST \
model.test_ds.manifest_filepath=$TEST_MANIFEST \
trainer.max_epochs=$MAX_EPOCHS \
model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
model.train_ds.bucketing_batch_size=$BUCKET_BATCH_SIZE \
model.train_ds.bucketing_strategy=$BUCKET_STRATEGY \
model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
model.test_ds.batch_size=$EVAL_BATCH_SIZE \
trainer.accumulate_grad_batches=${GRAD_ACC} \
trainer.gradient_clip_val=$GRAD_CLIP \
model.spec_augment.time_masks=$TIME_MASKS \
model.spec_augment.time_width=$TIME_WIDTH \
model.optim_param_groups.encoder.lr=$LR_ENC \
model.optim.name=$OPT \
model.optim.lr=$LR \
model.optim.weight_decay=$WD \
model.optim.sched.warmup_steps=$WARMUP \
model.optim.sched.name=$SCHEDULER \
model.decoder.num_layers=$DECODER_LAYERS \
model.decoder.inner_size=$DECODER_INNER_SIZE \
model.decoder.ffn_dropout=$DECODER_FFT_DROPOUT \
model.decoder.attn_score_dropout=$DECODER_ATTN_SCORE_DROPOUT \
model.decoder.attn_layer_dropout=$DECODER_ATTN_LAYER_DROPOUT \
model.ssl_pretrained.freeze=$FREEZE_ENCODER
EOF


srun -J $EXP_NAME -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"
set +
