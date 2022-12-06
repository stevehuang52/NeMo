# python audio_to_rttm.py manifest_filepath=/media/data/datasets/LibriSpeech/test_other_abs.json output_dir=vad_output_ls_other
# python audio_to_rttm.py manifest_filepath=/media/data/datasets/asr_vad_eval/manifests/english_test_min10.json output_dir=vad_output_min10
# python audio_to_rttm.py manifest_filepath=/media/data/datasets/asr_vad_eval/chris_data/chris.json

# python audio_to_rttm_frame.py manifest_filepath=./manifests_noise/musan_test.json output_dir=vad_output_frame_musan
MODEL_PATH=/home/heh/checkpoints/frame_vad_marblenet_ep50_ns_wce.nemo
# python audio_to_rttm_frame.py vad.model_path=$MODEL_PATH manifest_filepath=/media/data/datasets/asr_vad_eval/manifests/english_test_min10.json output_dir=vad_output_frame_min10
python audio_to_rttm_frame.py vad.model_path=$MODEL_PATH manifest_filepath=./manifests_noise/musan_test.json output_dir=vad_output_frame_musan
