import argparse
from ast import parse
import json
from pathlib import Path
from copy import deepcopy

def load_manifest(filepath):
    data = []
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            data.append(json.loads(line.strip()))
    return data

def save_manifest(data, filepath):
    with Path(filepath).open("w") as fout:
        for item in data:
            fout.write(f"{json.dumps(item)}\n")
    

def change_key(data, src_key, tgt_key):
    results = []
    for item in data:
        item = deepcopy(item)
        if src_key not in item:
            for key in item.keys():
                if src_key in key:
                    src_key = key
                    break
        
        item[tgt_key] = item[src_key]
        item.pop(src_key)
        results.append(item)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="path to manifest to be processed")
    parser.add_argument("-s", "--source", default="energy_vad_mask", help="the original key you want to change")
    parser.add_argument("-t", "--target", default="label", help="the target key you want to change to")
    parser.add_argument("-o", "--output", default="", help="path for output file")
    args = parser.parse_args()

    filepath = Path(args.manifest)
    if filepath.is_dir():
        manifest_list = list(filepath.glob("*.json"))
        print(f"Found {len(manifest_list)} files to be processed.")
    else:
        manifest_list = [filepath]
    
    for manifest in manifest_list:
        print(f"Processing: {manifest}")
        data = load_manifest(manifest)
        new_data = change_key(data, args.source, args.target)

        if not args.output:
            args.output = manifest
        print(f"Saving output to: {args.output}")
        save_manifest(new_data, args.output)

    print("Done!")