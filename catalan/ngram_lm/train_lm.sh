kenlm_bin_path="/home/heh/github/NeMo/catalan/ngram_lm/decoders/kenlm/build/bin"

mkdir -p kenlm_saved

N=6
tag="spu1024"
train_file="/home/heh/datasets/Catalan/catalan_data/train.json"

proj_name="ConformerL_RNNT_Catalan"
exp_name="drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f8_gacc1_ep1000_dgx1"

nemo_model_file="../results/${proj_name}/${exp_name}/${exp_name}-averaged.nemo"

kenlm_model_file="kenlm_saved/kenlm_N${N}_${tag}.model"

python train_kenlm.py \
    --nemo_model_file ${nemo_model_file} \
    --train_file ${train_file} \
    --kenlm_bin_path ${kenlm_bin_path} \
    --kenlm_model_file ${kenlm_model_file} \
    --ngram_length ${N}


# ConformerL_RNNT_Catalan
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4_b32
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f8_gacc1_ep1000_dgx1

# ConformerL_ctc_catalan
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32
