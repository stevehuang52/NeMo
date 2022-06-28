# Code adapted from https://github.com/speechbrain/speechbrain/blob/develop/recipes/SLURP/prepare.py

import json
import os
from pathlib import Path
from typing import Optional

import jsonlines
import pandas as pd
from tqdm import tqdm


def prepare_SLURP(
    data_folder: str,
    save_folder: Optional[str] = None,
    slu_type: str = "direct",
    skip_prep: bool = False,
    join_semantics: bool = False,
    audio_prefix: str = "",
):
    """
    This function prepares the SLURP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to SLURP dataset.
    save_folder: path where to save the csv manifest files.
    slu_type : one of the following:

      "direct":{input=audio, output=semantics}
      "multistage":{input=audio, output=semantics} (using ASR transcripts in the middle)
      "decoupled":{input=transcript, output=semantics} (using ground-truth transcripts)

    train_splits : list of splits to be joined to form train .csv
    skip_prep: If True, data preparation is skipped.
    """
    if skip_prep:
        return

    vocab = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    if not save_folder:
        save_folder = data_folder
    elif not Path(save_folder).is_dir():
        Path(save_folder).mkdir(parents=True)

    splits = [
        "train_real",
        "train_synthetic",
        "devel",
        "test",
    ]
    id = 0
    for split in splits:
        new_filename = os.path.join(save_folder, split) + f"-{slu_type}.json"

        print("Preparing %s..." % new_filename)

        IDs = []
        slurp_id = []
        audio = []
        audio_format = []
        audio_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        jsonl_path = os.path.join(data_folder, split + ".jsonl")

        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                sid = obj["slurp_id"]
                scenario = obj["scenario"]
                action = obj["action"]
                sentence_annotation = obj["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                entities = []
                for slot in range(num_entities):
                    type = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[0].strip()
                    filler = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[1].strip()
                    entities.append({"type": type, "filler": filler})
                for recording in obj["recordings"]:
                    IDs.append(id)
                    slurp_id.append(sid)
                    if "synthetic" in split:
                        audio_folder = "slurp_synth/"
                    else:
                        audio_folder = "slurp_real/"

                    path = os.path.join(audio_prefix, audio_folder, recording["file"])

                    audio.append(path)
                    audio_format.append("flac")
                    audio_opts.append(None)

                    transcript_ = obj["sentence"]
                    if slu_type == "decoupled":
                        transcript_ = transcript_.upper()
                    # transcript_ = "".join([c for c in transcript_ if c in vocab])

                    transcript.append(transcript_)
                    transcript_format.append("string")
                    transcript_opts.append(None)

                    semantics_dict = {
                        "scenario": scenario,
                        "action": action,
                        "entities": entities,
                    }
                    if join_semantics:
                        semantics_ = str(semantics_dict).replace(
                            ",", "|"
                        )  # Commas in dict will make using csv files tricky; replace with pipe.
                    else:
                        semantics_ = semantics_dict
                    semantics.append(semantics_)
                    semantics_format.append("string")
                    semantics_opts.append(None)
                    id += 1

        df = pd.DataFrame(
            {"ID": IDs, "slurp_id": slurp_id, "audio": audio, "semantics": semantics, "transcript": transcript,}
        )

        with open(new_filename, "w") as fout:
            for idx in tqdm(range(len(df))):
                item = {
                    "id": str(df["ID"][idx]),
                    "slurp_id": str(df["slurp_id"][idx]),
                    "audio_filepath": df["audio"][idx],
                    "transcript": df["transcript"][idx],
                    "semantics": df["semantics"][idx],
                }
                fout.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    data_root = "/home/heh/datasets/slurp-speechbrain"
    prepare_SLURP(data_folder=data_root, save_folder="../manifests_slu2asr", join_semantics=True)
