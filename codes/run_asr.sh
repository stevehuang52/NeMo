python feature_rttm_to_text.py \
    pretrained_name="stt_en_conformer_ctc_large" \
    dataset_manifest=./vad_output_musan/manifest_vad_feat_rttm.json \
    num_workers=8 \
    normalize=post_norm \
    use_noise=True \
    use_rttm=True \
    use_feature=True

# model_path=/home/heh/checkpoints/Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-3.0_no_weight_decay_e100-averaged.nemo \
# pretrained_name="stt_en_conformer_ctc_large" \
# model_path=/home/heh/checkpoints/Citrinet_Aug_1024_Gamma_0-25_NeMo_ASRSET_2.0_e200.nemo \


