export DATA_ROOT="/home/heh/datasets/slurp_draco"
python ./process_asr_text_tokenizer.py \
  --manifest="${DATA_ROOT}/train_real_nlu_oracle.json,${DATA_ROOT}/train_synth_nlu_oracle.json" \
  --data_root="${DATA_ROOT}/tokenizers_nlu/" \
  --data_key="text" \
  --vocab_size=512 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log \
  --spe_bos \
  --spe_eos \
  --spe_pad
