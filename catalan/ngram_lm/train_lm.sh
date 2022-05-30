kenlm_bin_path="/home/heh/github/NeMo/catalan/ngram_lm/decoders/kenlm/build/bin"

mkdir -p kenlm_saved

N=6
tag="spu128"
train_file="/home/heh/datasets/Catalan/catalan_data/train.json"

proj_name="ConformerL_ctc_catalan"
exp_name="drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32"

nemo_model_file="../results/${proj_name}/${exp_name}/${exp_name}-averaged.nemo"

kenlm_model_file="kenlm_saved/kenlm_N${N}_${tag}.model"

python train_kenlm.py \
    --nemo_model_file ${nemo_model_file} \
    --train_file ${train_file} \
    --kenlm_bin_path ${kenlm_bin_path} \
    --kenlm_model_file ${kenlm_model_file} \
    --ngram_length ${N}
