
kenlm_model_file="./kenlm_saved/kenlm_N6_spu128.model"

proj_name="ConformerL_ctc_catalan_abl"
exp_name="drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32_dgx1_full_v1"

pred_output="./predictions/${proj_name}/${exp_name}"
mkdir -p $pred_output

nemo_model_file="../results/${proj_name}/${exp_name}/${exp_name}-averaged.nemo"

# test_manifest="/home/heh/datasets/Catalan/catalan_cleaned/dev/dev.json"
# python eval_beamsearch_ngram.py \
#     --nemo_model_file $nemo_model_file \
#     --input_manifest $test_manifest \
#     --kenlm_model_file $kenlm_model_file \
#     --acoustic_batch_size 128 \
#     --beam_width 128 \
#     --beam_alpha 0.5 0.75 1.0 1.25 1.5 2.0 \
#     --beam_beta  0.5 0.75 1.0 1.25 1.5 2.0 \
#     --preds_output_folder $pred_output \
#     --decoding_mode beamsearch_ngram

test_manifest="/home/heh/datasets/Catalan/catalan_cleaned/test/test_v1.json"
CUDA_VISIBLE_DEVICES=0 python eval_beamsearch_ngram.py \
    --nemo_model_file $nemo_model_file \
    --input_manifest $test_manifest \
    --kenlm_model_file $kenlm_model_file \
    --acoustic_batch_size 128 \
    --beam_width 128 \
    --beam_alpha 1.5 \
    --beam_beta  2.0 \
    --preds_output_folder $pred_output \
    --decoding_mode beamsearch_ngram

# ConformerL_RNNT_Catalan
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4_b32
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f8_gacc1_ep1000_dgx1

# ConformerL_ctc_catalan
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32

# ConformerL_ctc_catalan_v2
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000_dgx2

# ConformerL_ctc_catalan_abl
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32_dgx1_full_v1

# ConformerL_RNNT_Catalan_abl
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_bk4_full_dgx1_v1
