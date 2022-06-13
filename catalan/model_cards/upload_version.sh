ngc registry model upload-version \
  --link "https://github.com/NVIDIA/NeMo" \
  --link-type GitHub \
  --metrics-file model_table_rnnt.json \
  --metrics-file accuracy_table_rnnt.json \
  --source stt_ca_conformer_transducer_large.nemo \
  nvstaging/nemo/stt_ca_conformer_transducer_large:1.11.0
