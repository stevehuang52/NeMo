curr_dir=${pwd}

proj_name="ASR_Finetune"
exp_dir="drc_ConfLCTC_SLURP_adamwlr1e-4_wd1e-3_gc0.0_CosineAnnealing_wp2000_aug10x0.05_b16_ep50_r2_dgx1"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

cp checkpoint_averaging.py ${target_dir}/

cd ${target_dir}
find . -name '*.nemo' | grep -v -- "-averaged.nemo" | xargs python checkpoint_averaging.py
