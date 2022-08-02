
SCRIPT="../examples/asr/speech_to_text_eval.py"
DATA_DIR="/home/heh/datasets/slurp_draco"

SPLIT="test"
DST="nlu_ft"
SRC="asr"
MANIFEST_FILE="${DATA_DIR}/${SPLIT}_${SRC}.json"
OUTPUT_FILE="manifests_nlu/${SPLIT}_${DST}.json"

# model_path="/media/data/pretrained/stt_en_conformer_ctc_large_slurp.nemo"
# pretrained_name=stt_en_conformer_transducer_large \

CUDA_VISIBLE_DEVICES=1 python $SCRIPT \
    model_path="/media/data/pretrained/stt_en_conformer_ctc_large_slurp.nemo" \
    dataset_manifest=$MANIFEST_FILE \
    output_filename=$OUTPUT_FILE \
    batch_size=32 \
    amp=True \
    use_cer=False
