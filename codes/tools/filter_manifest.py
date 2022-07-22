import argparse
import json
from pathlib import Path


def load_manifest(filepath: Path) -> list:
    data = []
    with filepath.open("r") as f:
        for line in f:
            datum = json.loads(line)
            data.append(datum)
    return data


def save_manifest(data: list, filepath: Path) -> None:
    with filepath.open("w") as f:
        for datum in data:
            line = json.dumps(datum)
            f.write(line + "\n")


def filter_data(data: list, text_key: str) -> list:
    new_data = []
    for datum in data:
        d = datum[text_key]
        if not d:
            continue
        new_data.append(datum)
    return new_data


def clean_manifest_text(filepath: str, text_key: str = "text", posfix: str = "cleaned"):
    print(f"Cleaning manifest: {filepath}")
    filepath = Path(filepath)
    outfile = Path(filepath.parent) / Path(filepath.stem + f"_{posfix}.json")
    data = load_manifest(filepath)
    new_data = filter_data(data, text_key)
    save_manifest(new_data, outfile)
    print(f"Done, {len(data) - len(new_data)} entries discarded, output is saved to: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=str, help="file path to manifest")
    parser.add_argument("-k", "--key", default="pred_text", type=str, help="field name for transcript in the manifest")
    parser.add_argument("-p", "--posfix", default="cleaned", type=str, help="posfix added to the output manifest")
    args = parser.parse_args()
    clean_manifest_text(args.manifest, args.key, args.posfix)
