# export DATA_ROOT="/home/heh/datasets/slurp-speechbrain"
# --manifest="${DATA_ROOT}/manifests/train_real-direct_decoded.json,${DATA_ROOT}/manifests/train_synthetic-direct_decoded.json" \
#   --data_root="${DATA_ROOT}/tokenizers_s2s/" \
DATA_ROOT="/home/heh/datasets/slurp_draco"
python ./process_asr_text_tokenizer.py \
  --manifest="${DATA_ROOT}/train_real_slu2asr.json,${DATA_ROOT}/manifests/train_synth_slu2asr.json" \
  --data_root="${DATA_ROOT}/tokenizers_slu2asr/" \
  --data_key="text" \
  --vocab_size=1024 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log \
  --spe_bos \
  --spe_eos \
  --spe_pad
