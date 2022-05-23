export CAT_DATA_ROOT="/home/heh/datasets/Catalan/catalan_data"
python ./process_asr_text_tokenizer.py \
  --manifest="${CAT_DATA_ROOT}/manifest/train.json" \
  --data_root="${CAT_DATA_ROOT}/tokenizers/" \
  --vocab_size=128 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log




