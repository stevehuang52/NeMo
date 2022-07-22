import argparse
import json
from pathlib import Path

alphabet = [' ','s', 'e', 'r', 'v', 'i', 'd', 'p', 'o', 'g', 'a', 'm', 't', 'u', 'l', 'f', 'c', 'z', 'b',
            'q', 'n', 'é', "'", 'x', 'ó', 'è', 'h', 'í', 'ü', 'j', 'à', 'ï', 'w', 'k', 'y', 'ç', 'ú', '-',
            'ò', 'á', '·', 'ñ', '—', '–', 'ı']

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

def remove_punc(data: list, text_key: str) -> list:
    new_data = []
    for datum in data:
        datum[text_key] = "".join([c for c in datum[text_key] if c in alphabet])
        new_data.append(datum)
    return new_data

def clean_manifest_text(filepath: str, text_key: str="text", posfix: str="cleaned"):
    print(f"Cleaning manifest: {filepath}")
    filepath = Path(filepath)
    outfile = Path(filepath.stem + f"_{posfix}.json")
    data = load_manifest(filepath)
    new_data = remove_punc(data, text_key)
    save_manifest(new_data, outfile)
    print(f"Done, output is saved to: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=str, help="file path to manifest")
    parser.add_argument("-k", "--key", default="text", type=str, help="field name for transcript in the manifest")
    parser.add_argument("-p", "--posfix", default="cleaned", type=str, help="posfix added to the output manifest")
    args = parser.parse_args()
    clean_manifest_text(args.manifest, args.key, args.posfix)

