import argparse
import json
import multiprocessing
import os
from pathlib import Path

import sox
import tqdm
from sox import Transformer
from tqdm.contrib.concurrent import process_map

FIELD_AUDIO = "audio_filepath"
FIELD_TEXT = "text"
FIELD_DATA_DIR = "data_dir"


def process(x):
    if not isinstance(x[FIELD_TEXT], str):
        x[FIELD_TEXT] = ''
    else:
        x[FIELD_TEXT] = x[FIELD_TEXT].lower().strip()

    data_dir = x[FIELD_DATA_DIR]
    input_file = Path(x[FIELD_AUDIO])
    if not input_file.is_absolute():
        input_file_path = data_dir / input_file
    else:
        input_file_path = input_file

    output_file = Path(input_file.stem + ".wav")

    if "slurp_real" in str(input_file_path):
        output_dir = Path("wavs/slurp_real")
    else:
        output_dir = Path("wavs/slurp_synth")

    output_file_path = str(data_dir / output_dir / output_file)

    if not os.path.exists(output_file_path):
        tfm = Transformer()
        tfm.rate(samplerate=16000)
        tfm.channels(n_channels=1)
        tfm.build(input_filepath=str(input_file_path), output_filepath=str(output_file_path))
    x['duration'] = sox.file_info.duration(str(output_file_path))
    x[FIELD_AUDIO] = str(output_dir / output_file)
    del x[FIELD_DATA_DIR]
    return x


def load_data(manifest, data_dir):
    data = []
    with open(manifest, 'r') as f:
        for line in tqdm.tqdm(f):
            item = json.loads(line)
            item[FIELD_DATA_DIR] = Path(data_dir)
            data.append(item)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest', default="/home/heh/datasets/slurp-speechbrain/manifests", help='path to the original manifest'
    )
    parser.add_argument('--data_dir', default="/home/heh/datasets/slurp-speechbrain", help='path to the data dir')
    parser.add_argument(
        "--num_workers", default=multiprocessing.cpu_count(), type=int, help="Workers to process dataset."
    )
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    wavs_dir = Path(args.data_dir) / Path("wavs")
    wavs_dir.mkdir(parents=True, exist_ok=True)
    wavs_real_dir = wavs_dir / Path("slurp_real")
    wavs_real_dir.mkdir(parents=True, exist_ok=True)
    wavs_synth_dir = wavs_dir / Path("slurp_synth")
    wavs_synth_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if manifest_path.is_dir():
        manifest_list = list(manifest_path.glob("*.jsonl"))
    else:
        manifest_list = [str(manifest_path)]

    print(f"Found {len(manifest_list)} manifests to be processed.")
    for manifest in manifest_list:
        print(f"Processing manifest: {manifest}")
        ext = manifest.suffix
        data = load_data(str(manifest), args.data_dir)
        if args.single:
            data_new = []
            for datum in tqdm.tqdm(data):
                data_new.append(process(datum))
        else:
            data_new = process_map(process, data, max_workers=args.num_workers, chunksize=100)

        with open(str(manifest).replace(ext, f'_decoded{ext}'), 'w') as f:
            for item in tqdm.tqdm(data_new):
                f.write(json.dumps(item) + '\n')
