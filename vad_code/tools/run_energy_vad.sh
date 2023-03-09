manifest_dir="/media/data/projects/NeMo-fvad/vad_code/manifests_cleaned/train"
out_dir=/media/data2/datasets/multilang_vad_ctm

# manifest_list = [
#     manifest_dir / "french_train_40ms_local_cleaned.json",
#     manifest_dir / "german_train_40ms_local_cleaned.json",
#     manifest_dir / "mandarin_train_40ms_local_cleaned.json",
#     manifest_dir / "russian_train_40ms_local_cleaned.json",
#     manifest_dir / "spanish_train_40ms_local_cleaned.json"
# ]

split="mandarin"
manifest_file="$manifest_dir/${split}_train_40ms_local_cleaned.json"
out_dir=$out_dir/${split}

python energy_vad.py --manifest_file $manifest_file --out_dir $out_dir --threshold 5 --enable_ctm

