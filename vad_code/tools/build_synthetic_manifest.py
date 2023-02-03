import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_text_file(filepath: str) -> List[str]:
    results = []
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            results.append(line)
    return results


def read_manifest(filepath: str) -> List[dict]:
    results = []
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def write_manifest(manifest: List[dict], filepath: str) -> None:
    with Path(filepath).open("w") as fout:
        for item in manifest:
            fout.write(f"{json.dumps(item)}\n")


def load_rttm_file(filepath: str) -> dict:
    data = pd.read_csv(filepath, sep="\s+", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    data['start'] = data['start'].astype(float)
    data['dur'] = data['dur'].astype(float)
    data['end'] = data['start'] + data['dur']

    data = data.sort_values(by=['start'])
    data['segment'] = list(zip(data['start'], data['end']))

    return data


def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def get_speech_segments(rttm_file: str) -> List[List[float]]:
    speech_segments = list(load_rttm_file(rttm_file)['segment'])
    speech_segments = [list(x) for x in speech_segments]
    speech_segments = merge_intervals(speech_segments)
    return speech_segments


def get_frame_labels(segments: List[List[float]], frame_length: float, offset: float, duration: float) -> str:
    labels = []
    n_frames = int(np.ceil(duration / frame_length))
    sid = 0
    for i in range(n_frames):
        t = offset + i * frame_length
        while sid < len(segments) - 1 and segments[sid][1] < t:
            sid += 1
        if segments[sid][0] <= t <= segments[sid][1]:
            labels.append('1')
        else:
            labels.append('0')
    return ' '.join(labels)


def generate_manifest_entry(audio_filepath):
    vad_frame_unit_secs = 0.08
    audio_filepath = Path(audio_filepath)
    y, sr = librosa.load(str(audio_filepath))
    dur = librosa.get_duration(y=y, sr=sr)

    manifest_path = audio_filepath.parent / Path(f"{audio_filepath.stem}.json")
    audio_manifest = read_manifest(manifest_path)
    text = " ".join([x["text"] for x in audio_manifest])

    rttm_path = audio_filepath.parent / Path(f"{audio_filepath.stem}.rttm")
    segments = get_speech_segments(rttm_path)
    labels = get_frame_labels(segments, vad_frame_unit_secs, 0.0, dur)

    entry = {
        "audio_filepath": str(audio_filepath.absolute()),
        "offset": 0.0,
        "duration": dur,
        "text": text,
        "label": labels,
        "vad_frame_unit_secs": vad_frame_unit_secs,
    }
    return entry


def main(args):
    wav_list = read_text_file(Path(args.input_dir, "synthetic_wav.list"))
    print(f"Found {len(wav_list)} in directory: {args.input_dir}")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        manifest_data = list(tqdm(pool.imap(generate_manifest_entry, wav_list), total=len(wav_list)))

    write_manifest(manifest_data, args.output_file)
    print(f"Manifest saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", default=None, help="Path to directory containing synthetic data")
    parser.add_argument("-o", "--output_file", default=None, help="Path to output manifest file")

    args = parser.parse_args()
    main(args)
