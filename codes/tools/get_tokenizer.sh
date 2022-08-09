export DATA_ROOT="/home/heh/datasets/slurp-speechbrain"
python ./process_asr_text_tokenizer.py \
  --manifest="${DATA_ROOT}/manifests/train_real-direct_decoded.json,${DATA_ROOT}/manifests/train_synthetic-direct_decoded.json" \
  --data_root="${DATA_ROOT}/tokenizers_s2s/" \
  --data_key="semantics" \
  --vocab_size=58 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log \
  --spe_bos \
  --spe_eos \
  --spe_pad
