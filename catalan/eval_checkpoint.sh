curr_dir=${pwd}
test_manifest="/home/heh/datasets/Catalan/catalan_cleaned/test/test.json"

proj_name="ConformerL_ctc_catalan_v2"
exp_name="drc_catalan_d512_adamwlr2.0_wd1e-3_aug10x0.05_spu128_emit_bn_b32_f_gacc1_ep1000_dgx2"

target_dir=./results/${proj_name}/${exp_name}/
nemo_file=${exp_name}-averaged.nemo


cp speech_to_text_eval.py ${target_dir}/
cp transcribe_speech.py ${target_dir}/
cp transcribe_speech_parallel.py ${target_dir}/

cd ${target_dir}

python speech_to_text_eval.py \
    model_path=${nemo_file} \
    dataset_manifest=${test_manifest} \
    output_filename="evaluation_transcripts.json" \
    batch_size=32 \
    amp=True \
    use_cer=False

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
