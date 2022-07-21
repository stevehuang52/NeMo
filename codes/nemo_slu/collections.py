import collections
import json
import os
from os.path import expanduser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.utils import logging


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class _AudioTextSemantics(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextSemanticsEntity',
        field_names='id audio_file duration text_tokens semantic_tokens pred_text_tokens offset text_raw semantics_raw pred_text_raw speaker orig_sr lang',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        semantics: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        text_tokens: List[Optional[int]],
        semantic_tokens: List[Optional[int]],
        langs: List[Optional[str]],
        pred_text: List[Optional[str]],
        pred_text_tokens: List[Optional[str]],
        text_parser: Optional[parsers.CharParser] = None,
        semantic_parser: Optional[parsers.CharParser] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            semantics: List of SLU semantic annotations.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            text_tokens: List of integer tokens for audio transcripts,
            semantic_tokens: List of integer tokens for semantics,
            langs: List of language ids, one for eadh sample, or None.
            text_parser: Instance of `CharParser` to convert string to tokens.
            semantic_parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for (
            id_,
            audio_file,
            duration,
            offset,
            text,
            semantic,
            p_text,
            speaker,
            orig_sr,
            text_token,
            semantic_token,
            p_text_tokens,
            lang,
        ) in zip(
            ids,
            audio_files,
            durations,
            offsets,
            texts,
            semantics,
            pred_text,
            speakers,
            orig_sampling_rates,
            text_tokens,
            semantic_tokens,
            pred_text_tokens,
            langs,
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            # tokenize text
            if text_token is not None:
                _text_tokens = text_token
            else:
                if text != '' and text_parser:
                    if hasattr(text_parser, "is_aggregate") and text_parser.is_aggregate:
                        if lang is not None:
                            _text_tokens = text_parser(text, lang)
                        else:
                            raise ValueError("lang required in manifest when using aggregate tokenizers")
                    else:
                        _text_tokens = text_parser(text)
                else:
                    _text_tokens = []

                if _text_tokens is None:
                    duration_filtered += duration
                    num_filtered += 1
                    continue

            if p_text_tokens is not None:
                _pred_text_tokens = p_text_tokens
            else:
                if p_text and text_parser:
                    if hasattr(text_parser, "is_aggregate") and text_parser.is_aggregate:
                        if lang is not None:
                            _pred_text_tokens = text_parser(p_text, lang)
                        else:
                            raise ValueError("lang required in manifest when using aggregate tokenizers")
                    else:
                        _pred_text_tokens = text_parser(p_text)
                else:
                    _pred_text_tokens = []

            # tokenize semantics
            if semantic_token is not None:
                _semantic_tokens = semantic_token
            elif semantic != '' and semantic_parser:
                if hasattr(semantic_parser, "is_aggregate") and semantic_parser.is_aggregate:
                    if lang is not None:
                        _semantic_tokens = semantic_parser(semantic, lang)
                    else:
                        raise ValueError("lang required in manifest when using aggregate tokenizers")
                else:
                    _semantic_tokens = semantic_parser(semantic)
            else:
                _semantic_tokens = []

            total_duration += duration

            data.append(
                output_type(
                    id_,
                    audio_file,
                    duration,
                    _text_tokens,
                    _semantic_tokens,
                    _pred_text_tokens,
                    offset,
                    text,
                    semantic,
                    p_text,
                    speaker,
                    orig_sr,
                    lang,
                )
            )
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                if file_id not in self.mapping:
                    self.mapping[file_id] = []
                self.mapping[file_id].append(len(data) - 1)

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class AudioTextSemantics(_AudioTextSemantics):
    """`AudioTextSemantics` collector from SLU structured json files."""

    FIELD_ID = "id"
    FIELD_AUDIO = "audio_file"
    FIELD_DURATION = "duration"
    FIELD_TEXT = "text"
    FIELD_SEMANTICS = "semantics"
    FIELD_OFFSET = "offset"
    FIELD_SPEAKER = "speaker"
    FIELD_ORSR = "orig_sr"
    FIELD_LANG = "lang"
    FIELD_TEXT_TOKENS = "text_tokens"
    FIELD_SEMANTIC_TOKENS = "semantic_tokens"
    FIELD_PRED_TEXT = "pred_text"
    FIELD_PRED_TEXT_TOKENS = "pred_text_tokens"

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations, transcripts texts and semantics for SLU.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioTextSemantics` constructor.
            **kwargs: Kwargs to pass to `AudioTextSemantics` constructor.
        """

        ids, audio_files, durations, offsets, = [], [], [], []
        texts, text_tokens, semantics, semantic_tokens = [], [], [], []
        speakers, orig_srs, langs = [], [], []
        pred_text, pred_text_tokens = [], []
        for item in manifest.item_iter(manifests_files, parse_func=self.__slu_parse_item):
            ids.append(item[self.FIELD_ID])
            audio_files.append(item[self.FIELD_AUDIO])
            durations.append(item[self.FIELD_DURATION])
            texts.append(item[self.FIELD_TEXT])
            semantics.append(item[self.FIELD_SEMANTICS])
            offsets.append(item[self.FIELD_OFFSET])
            speakers.append(item[self.FIELD_SPEAKER])
            orig_srs.append(item[self.FIELD_ORSR])
            text_tokens.append(item[self.FIELD_TEXT_TOKENS])
            semantic_tokens.append(item[self.FIELD_SEMANTIC_TOKENS])
            langs.append(item[self.FIELD_LANG])
            pred_text.append(item[self.FIELD_PRED_TEXT])
            pred_text_tokens.append(item[self.FIELD_PRED_TEXT_TOKENS])

        super().__init__(
            ids,
            audio_files,
            durations,
            texts,
            semantics,
            offsets,
            speakers,
            orig_srs,
            text_tokens,
            semantic_tokens,
            langs,
            pred_text,
            pred_text_tokens,
            *args,
            **kwargs,
        )

    def __slu_parse_item(cls, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        if "index" in item:
            item[cls.FIELD_ID] = item.pop("index")

        # Audio file
        if 'audio_filename' in item:
            item[cls.FIELD_AUDIO] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item[cls.FIELD_AUDIO] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
            )

        # If the audio path is relative, and not using tarred dataset,
        # attach the parent directory of manifest to the audio path.
        # Assume "audio_file" starts with a dir, such as "wavs/xxxxx.wav".
        # If using a tarred dataset, the "audio_path" is like "_home_data_tarred_wavs_xxxx.wav",
        # so we will just ignore it.
        manifest_dir = Path(manifest_file).parent
        audio_file = Path(item[cls.FIELD_AUDIO])
        if not audio_file.is_file() and not audio_file.is_absolute() and audio_file.parent != Path("."):
            # assume the wavs/ dir and manifest are under the same parent dir
            audio_file = manifest_dir / audio_file
            item[cls.FIELD_AUDIO] = str(audio_file.absolute())
        else:
            item[cls.FIELD_AUDIO] = expanduser(item[cls.FIELD_AUDIO])

        # Duration.
        if cls.FIELD_DURATION not in item:
            raise ValueError(
                f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
            )

        # Text.
        if cls.FIELD_TEXT in item:
            pass
        elif 'text_filepath' in item:
            with open(item.pop('text_filepath'), 'r') as f:
                item[cls.FIELD_TEXT] = f.read().replace('\n', '')
        elif 'normalized_text' in item:
            item[cls.FIELD_TEXT] = item['normalized_text']

        # Semantics
        if "semantics" in item:
            key_semantic = "semantics"
        elif "semantic" in item:
            key_semantic = "semantic"
        else:
            raise ValueError("Cannot find key for semantics/semantic in manifest")
        item[cls.FIELD_SEMANTICS] = str(item.pop(key_semantic)).replace(",", "|")  # make compatible with speechbrain

        # prepare data for return
        datum = dict()
        datum[cls.FIELD_AUDIO] = item[cls.FIELD_AUDIO]
        datum[cls.FIELD_DURATION] = item[cls.FIELD_DURATION]
        datum[cls.FIELD_TEXT] = item.get(cls.FIELD_TEXT, "")
        datum[cls.FIELD_SEMANTICS] = item.get(cls.FIELD_SEMANTICS, "")
        datum[cls.FIELD_OFFSET] = item.get(cls.FIELD_OFFSET, None)
        datum[cls.FIELD_SPEAKER] = item.get(cls.FIELD_SPEAKER, None)
        datum[cls.FIELD_ORSR] = item.get(cls.FIELD_ORSR, None)
        datum[cls.FIELD_TEXT_TOKENS] = item.get(cls.FIELD_TEXT_TOKENS, None)
        datum[cls.FIELD_SEMANTIC_TOKENS] = item.get(cls.FIELD_SEMANTIC_TOKENS, None)
        datum[cls.FIELD_LANG] = (item.get(cls.FIELD_LANG, None),)
        datum[cls.FIELD_PRED_TEXT] = item.get(cls.FIELD_PRED_TEXT, None)
        datum[cls.FIELD_PRED_TEXT_TOKENS] = item.get(cls.FIELD_PRED_TEXT_TOKENS, None)
        return datum
