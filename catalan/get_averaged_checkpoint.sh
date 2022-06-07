curr_dir=${pwd}

proj_name="ConformerL_ctc_catalan_v2"
exp_dir="drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000_dgx2"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./results/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

cp checkpoint_averaging.py ${target_dir}/

cd ${target_dir}
find . -name '*.nemo' | grep -v -- "-averaged.nemo" | xargs python checkpoint_averaging.py

# ConformerL_RNNT_Catalan
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4_b32
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f8_gacc1_ep1000_dgx1

# ConformerL_ctc_catalan
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32


# ConformerL_RNNT_Catalan_v2
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f16_gacc1_ep1000_dgx2

# ConformerL_ctc_catalan_v2
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000_dgx2
