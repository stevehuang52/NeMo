#!/bin/bash
#SBATCH -A ent_aiapps_asr
#SBATCH -p batch_dgx2h_m2       # batch / batch_short / backfill
#SBATCH -N 8                   # number of nodes (32 nodes max for draco)
#SBATCH -t 8:00:00              # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --gpus-per-node=16      # n gpus per machine <required>
#SBATCH --ntasks-per-node=16    # n tasks per machine (one task per gpu) <required>
#SBATCH --nv-meta=ml-model.CONFORMERRNNT

set -x

####### Common Constants ######
SLURM_ACCOUNT='ent_aiapps'
USERID='users/heh'
WANDB_TOKEN="f5029311df02a27459c2c99c5fbef08978dc709e"
LUSTRE_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}
CLUSTER='drc'
###############################

# Container to be used by job. TIP: create "complete" containers, do not do pip/apt install as part of job
CONTAINER="gitlab-master.nvidia.com/heh/nemo_containers:nemo-22may16"

#### Project Constants ####
PROJECT_NAME=conformer_rnnt_L_catalan
DATASET=catalan
###########################

##### Code&Config Location ####
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/code/nemo-dev
CONFIG_PATH='../conf/conformer/'
CONFIG_NAME=conformer_transducer_${ENCODING}
###############################

#### Model Initialization ####
INIT_EXP=''
INIT_MODEL=''
#INIT_MODEL='./conformer_ls_rnnt_m/rno_ls_d256_adamwlr5.0_wd1e-2_aug10x0.05_wpe1024_emit1e-3_toktrue/rno_ls_d256_adamwlr5.0_wd1e-2_aug10x0.05_wpe1024_emit1e-3_toktrue/checkpoints/epoch215.ckpt'
#INIT_MODEL=stt_en_conformer_transducer_large
##############################

#### Hyper-Parameters ##########
GRAD_ACC=1
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=8
BUCKET_BATCH_SIZE=[40,32,24,16]
MAX_EPOCHS=500
SAVE_TOP_K=10
LOG_PREDICTION=false

# Conformer Params
D_MODEL=512
N_LAYERS=17
CONV_SIZE=31
SAMPLING_FACTOR=4
N_HEADS=8
TIME_MASKS=10
TIME_WIDTH=0.05
VOCAB_SIZE=1024
ENCODING=bpe # char or bpe
TOKENIZER_TYPE=bpe #bpe or wpe
OPT=adamw
LR=2.0
WD=1e-3
# Transducer Params
START_END_TOKEN=false
DECODER_SIZE=640
FUSED_BATCH_SIZE=2  # increase it as much as it works
EMIT_LAMBDA=0
# Misc
MODEL_POSFIX=''
###############################

##### Dataset ####
TOKENIZER=/tokenizers/NeMo_ASR_SET/English/asr_set_3.0/tokenizer_spe_unigram_v${VOCAB_SIZE}/
MAX_DURATION=16.0

TRAIN_WORKERS=2
TRAIN_ISTARRED=true
# Non-bucketing
TRAIN_MANIFEST='/data/librispeech_sp_tarred/tarred_audio_manifest.json'
TRAIN_FILEPATHS='/data/librispeech_sp_tarred/audio__OP_0..511_CL_.tar'
# Bucketing, set batch_size to 1, use BUCKET_BATCH_SIZE instead
#TRAIN_MANIFEST='[/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket1/tarred_audio_manifest.json,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket2/tarred_audio_manifest.json,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket3/tarred_audio_manifest.json,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket4/tarred_audio_manifest.json]'
#TRAIN_FILEPATHS='[/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket1/audio__OP_0..4095_CL_.tar,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket2/audio__OP_0..4095_CL_.tar,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket3/audio__OP_0..4095_CL_.tar,/data/nemo_asr_set/nemo_asrset_2.0_bucketing/bucket4/audio__OP_0..4095_CL_.tar]'

VAL_WORKERS=4
VAL_ISTARRED=false
VAL_MANIFEST='[/manifests/librispeech/librivox-dev-other.json,/manifests/librispeech/librivox-dev-clean.json,/manifests/librispeech/librivox-test-other.json,/manifests/librispeech/librivox-test-clean.json]'
VAL_FILEPATHS='na'

TEST_WORKERS=4
TEST_ISTARRED=false
TEST_MANIFEST='[/manifests/librispeech/librivox-dev-other.json,/manifests/librispeech/librivox-dev-clean.json,/manifests/librispeech/librivox-test-other.json,/manifests/librispeech/librivox-test-clean.json]'
TEST_FILEPATHS='na'
##################




################ Usually No Change #########################################################################
### Experiment Name ###
EXP_NAME=${CLUSTER}_${DATASET}_d${D_MODEL}_${OPT}lr${LR}_wd${WD}_aug${TIME_MASKS}x${TIME_WIDTH}spu${VOCAB_SIZE}_emit${EMIT_LAMBDA}_bn_b${TRAIN_BATCH_SIZE}_gacc${GRAD_ACC}_ep${MAX_EPOCHS}${MODEL_POSFIX}

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

if [ $ENCODING == 'char' ]
then
  SCRIPT_POSTFIX=''
else
  SCRIPT_POSTFIX='_bpe'
fi

if [ -z "$INIT_MODEL" ]
then
  INIT_CHECKPOINT=''
else
  INIT_CHECKPOINT=/exp_init/$INIT_MODEL
  #CHECKPOINT_DIR=$INIT_EXP_DIR/$INIT_EXP/$INIT_EXP/checkpoints/
  #INIT_CHECKPOINT=/exp_init/$INIT_EXP/$INIT_EXP/checkpoints/$(ls -l $CHECKPOINT_DIR | grep '\-last.ckpt' | awk '{print $9}')
fi

### Script to Run ###
SCRIPT_FILE="./examples/asr/asr_transducer/speech_to_text_rnnt$SCRIPT_POSTFIX.py"
############################################################################################################



#+init_from_pretrained_model=$INIT_MODEL \

# +init_from_nemo_model=$INIT_CHECKPOINT \
#+model.init_weights_from_model=$INIT_CHECKPOINT \
#conda install -c numba numba=0.53.1 -y


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
+trainer.benchmark=false \
trainer.precision=32 \
model.log_prediction=$LOG_PREDICTION \
model.train_ds.pin_memory=true \
model.test_ds.pin_memory=true \
model.validation_ds.pin_memory=true \
model.train_ds.use_start_end_token=$START_END_TOKEN \
model.validation_ds.use_start_end_token=$START_END_TOKEN \
model.test_ds.use_start_end_token=$START_END_TOKEN \
model.preprocessor.window_size=0.025 \
model.preprocessor.pad_to=0 \
model.train_ds.num_workers=$TRAIN_WORKERS \
model.validation_ds.num_workers=$VAL_WORKERS \
model.test_ds.num_workers=$TEST_WORKERS \
exp_manager.exp_dir=/results/ \
trainer.check_val_every_n_epoch=1 \
trainer.num_nodes=$SLURM_JOB_NUM_NODES \
model.tokenizer.type=$TOKENIZER_TYPE \
model.tokenizer.dir=$TOKENIZER \
model.train_ds.max_duration=${MAX_DURATION} \
model.train_ds.shuffle_n=2048 \
model.train_ds.trim_silence=false \
model.train_ds.is_tarred=$TRAIN_ISTARRED \
model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
model.validation_ds.manifest_filepath=$VAL_MANIFEST \
model.test_ds.manifest_filepath=$TEST_MANIFEST \
trainer.max_epochs=$MAX_EPOCHS \
model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
model.train_ds.bucketing_batch_size=$BUCKET_BATCH_SIZE \
model.train_ds.bucketing_strategy=synced_randomized \
model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
model.test_ds.batch_size=$EVAL_BATCH_SIZE \
trainer.accumulate_grad_batches=${GRAD_ACC} \
model.spec_augment.freq_masks=2 \
model.spec_augment.freq_width=27 \
model.spec_augment.time_masks=$TIME_MASKS \
model.spec_augment.time_width=$TIME_WIDTH \
model.encoder.d_model=$D_MODEL \
model.encoder.n_layers=$N_LAYERS \
model.encoder.n_heads=$N_HEADS \
model.encoder.conv_kernel_size=$CONV_SIZE \
model.encoder.self_attention_model=rel_pos \
model.encoder.subsampling=striding \
model.encoder.subsampling_conv_channels=$D_MODEL \
model.encoder.subsampling_factor=$SAMPLING_FACTOR \
model.encoder.dropout_emb=0.0 \
model.encoder.dropout_att=0.1 \
model.optim.name=$OPT \
model.optim.lr=$LR \
model.optim.betas=[0.9,0.98] \
model.optim.weight_decay=$WD \
model.optim.sched.warmup_steps=10000 \
model.optim.sched.name=NoamAnnealing \
model.optim.sched.d_model=$D_MODEL \
model.optim.sched.min_lr=1e-6 \
trainer.gradient_clip_val=0 \
model.compute_eval_loss=false \
model.decoding.strategy=greedy_batch \
model.decoder.prednet.pred_rnn_layers=1 \
model.decoder.prednet.dropout=0.1 \
model.model_defaults.pred_hidden=$DECODER_SIZE \
model.model_defaults.joint_hidden=$DECODER_SIZE \
model.decoder.prednet.pred_hidden=$DECODER_SIZE \
model.joint.jointnet.joint_hidden=$DECODER_SIZE \
model.joint.fuse_loss_wer=true \
model.joint.fused_batch_size=$FUSED_BATCH_SIZE \
model.joint.jointnet.activation=relu \
model.joint.jointnet.dropout=0.1
EOF


srun -J $EXP_NAME -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  $MOUNTS \
  bash -c "${cmd}"
set +