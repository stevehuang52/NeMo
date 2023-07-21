

NEMO_ROOT=/home/heh/codes/nemo-slm
export PYTHONPATH=${NEMO_ROOT}:${PYTHONPATH}

TP=1
PP=1

# INPUT_DIR=/media/data3/speech_llm/llama_hf/llama_hf/llama-13b-hf
# OUTPUT_DIR=/media/data3/speech_lm/llama/llama-13b-nemo

INPUT_DIR=/media/data3/pretrained_models/llama2_hf/llama-2-7b
OUTPUT_DIR=/media/data3/pretrained_models/llama2_nemo/llama2_7b/tp${TP}_pp${PP}/llama-2-7b.nemo

WORLD_SIZE=$TP python ../tools/convert_hf_llama_to_nemo.py --input_dir $INPUT_DIR  --output_file $OUTPUT_DIR --local_rank 0 --tensor_model_parallel_size $TP --pipeline_model_parallel_size $PP
