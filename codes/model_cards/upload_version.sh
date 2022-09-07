ngc registry model upload-version \
  --link "https://github.com/NVIDIA/NeMo" \
  --link-type GitHub \
  --metrics-file model_table_slurp.json \
  --metrics-file accuracy_table_slurp.json \
  --source ./checkpoints/slu_conformer_transformer_large_slurp.nemo \
  nvstaging/nemo/slu_conformer_transformer_large_slurp:1.13.0
