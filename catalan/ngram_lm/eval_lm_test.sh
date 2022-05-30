
pred_output="./predictions"
mkdir -p $pred_output

kenlm_model_file="./kenlm_saved/kenlm_N6_spu128.model"

proj_name="ConformerL_ctc_catalan"
exp_name="drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32"

nemo_model_file="../results/${proj_name}/${exp_name}/${exp_name}-averaged.nemo"
test_manifest="/home/heh/datasets/Catalan/catalan_cleaned/test/test.json"


python eval_beamsearch_ngram.py \
    --nemo_model_file $nemo_model_file \
    --input_manifest $test_manifest \
    --kenlm_model_file $kenlm_model_file \
    --acoustic_batch_size 128 \
    --beam_width 128 \
    --beam_alpha 1.0 \
    --beam_beta 1.0 \
    --preds_output_folder $pred_output \
    --decoding_mode beamsearch_ngram


# ConformerL_RNNT_Catalan
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b1_f8_gacc1_ep1000_dgx1_bk4_b32
# drc_catalan_d512_adamwlr5.0_wd1e-3_aug10x0.05_spu1024_emit0_bn_b32_f8_gacc1_ep1000_dgx1

# ConformerL_ctc_catalan
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000
# drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b1_f_gacc1_ep1000_bk4_b32
