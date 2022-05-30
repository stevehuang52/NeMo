# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from os.path import expanduser
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class ManifestBase:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


class ManifestEN:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


def item_iter(
    manifests_files: Union[str, List[str]],
    parse_func: Callable[[str, Optional[str]], Dict[str, Any]] = None,
    data_prefix: Optional[Union[str, List[str]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.

    NeMo ASR pipelines often assume certain manifest files structure. In
    particular, each manifest file should consist of line-per-sample files with
    each line being correct json dict. Each such json dict should have a field
    for audio file string, a field for duration float and a field for text
    string. Offset also could be additional field and is set to None by
    default.

    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.

        parse_func: A callable function which accepts as input a single line
            of a manifest and optionally the manifest file itself,
            and parses it, returning a dictionary mapping from str -> Any.

        data_prefix: Either single string prefix or list of such -
            prefix to append to each manifest.

    Yields:
        Parsed key to value item dicts.

    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    num_manifests = len(manifests_files)
    if isinstance(data_prefix, str):
        # use the same prefix for all manifests
        data_prefix = [data_prefix] if num_manifests == 1 else [data_prefix] * num_manifests
    elif not data_prefix:
        # append no prefix
        data_prefix = [""] * num_manifests

    num_prefix = len(data_prefix)
    if num_prefix > 1 and num_prefix != num_manifests:
        raise ValueError(
            f"The number of manifests and prefixes must match, unless a single or none prefix is provided."
            f"Got {num_manifests} manifests with {num_prefix} prefix(es)."
        )

    if parse_func is None:
        parse_func = __parse_item

    k = -1
    for manifest_file, prefix in zip(manifests_files, data_prefix):
        manifest_file = Path(manifest_file)
        if not manifest_file.is_file():
            manifest_file = Path(prefix) / Path(manifest_file)
        manifest_dir = Path(manifest_file.parent)
        with open(expanduser(str(manifest_file)), 'r') as f:
            for line in f:
                k += 1
                item = parse_func(line, manifest_file)
                audio_file = Path(prefix) / Path(item['audio_file'])
                if not audio_file.is_file():
                    audio_file = manifest_dir / audio_file  # assume wavs/ and manifest are under the same dir
                item['id'] = k
                item['audio_file'] = str(audio_file)
                yield item


# def item_iter(
#     manifests_files: Union[str, List[str]],
#     parse_func: Callable[[str, Optional[str]], Dict[str, Any]] = None,
# ) -> Iterator[Dict[str, Any]]:
#     """Iterate through json lines of provided manifests.

#     NeMo ASR pipelines often assume certain manifest files structure. In
#     particular, each manifest file should consist of line-per-sample files with
#     each line being correct json dict. Each such json dict should have a field
#     for audio file string, a field for duration float and a field for text
#     string. Offset also could be additional field and is set to None by
#     default.

#     Args:
#         manifests_files: Either single string file or list of such -
#             manifests to yield items from.

#         parse_func: A callable function which accepts as input a single line
#             of a manifest and optionally the manifest file itself,
#             and parses it, returning a dictionary mapping from str -> Any.

#     Yields:
#         Parsed key to value item dicts.

#     Raises:
#         ValueError: If met invalid json line structure.
#     """

#     if isinstance(manifests_files, str):
#         manifests_files = [manifests_files]

#     if parse_func is None:
#         parse_func = __parse_item

#     k = -1
#     for manifest_file in manifests_files:
#         manifest_file = Path(manifest_file)
#         manifest_dir = Path(manifest_file.parent)
#         with open(expanduser(str(manifest_file)), 'r') as f:
#             for line in f:
#                 k += 1
#                 item = parse_func(line, manifest_file)
#                 audio_file = Path(item['audio_file'])
#                 # if the audio filepath is relative, and not using tarred dataset
#                 # attach the parent directory of manifest to the audio paths if not tarrrd dataset
#                 if not audio_file.is_file() and audio_file.parent != Path("."):
#                     audio_file = manifest_dir / audio_file # assume the wavs/ dir and manifest are under the same dir
#                 item['id'] = k
#                 item['audio_file'] = str(audio_file)
#                 yield item


def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    item = json.loads(line)

    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
        )
    item['audio_file'] = expanduser(item['audio_file'])

    # Duration.
    if 'duration' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
        )

    # Text.
    if 'text' in item:
        pass
    elif 'text_filepath' in item:
        with open(item.pop('text_filepath'), 'r') as f:
            item['text'] = f.read().replace('\n', '')
    elif 'normalized_text' in item:
        item['text'] = item['normalized_text']

    item = dict(
        audio_file=item['audio_file'],
        duration=item['duration'],
        text=item.get('text', ""),
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        lang=item.get('lang', None),
    )

    return item
