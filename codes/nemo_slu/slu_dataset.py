import json
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, open_dict
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import ChainDataset

from . import collections
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common import tokenizers
from nemo.core.classes import Dataset
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = ["AudioTextSemanticsBPEDataset", "get_slu_bpe_dataset", "EnCharTokenizer"]


class DataConstantsSLU:
    FIELD_AUDIO = "audio_signal"
    FIELD_AUDIO_LEN = "audio_length"
    FIELD_TEXT = "text"
    FIELD_TEXT_LEN = "text_length"
    FIELD_SEMANTICS = "semantics"
    FIELD_SEMANTICS_LEN = "semantics_length"
    FIELD_SAMPLE_ID = "sample_id"
    FIELD_SENTIMENT = "sentiment"


class _AudioTextSemanticsDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'audio_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'semantics': NeuralType(('B', 'T'), LabelsType()),
            'semantics_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        text_parser: Union[str, Callable],
        semantic_parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ManifestProcessorSLU(
            manifest_filepath=manifest_filepath,
            text_parser=text_parser,
            semantic_parser=semantic_parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        s, sl = self.manifest_processor.process_semantics_by_sample(sample=sample)

        if self.return_sample_id:
            output = (
                f,
                fl,
                torch.tensor(t).long(),
                torch.tensor(tl).long(),
                torch.tensor(s).long(),
                torch.tensor(sl).long(),
                index,
            )
        else:
            output = (
                f,
                fl,
                torch.tensor(t).long(),
                torch.tensor(tl).long(),
                torch.tensor(s).long(),
                torch.tensor(sl).long(),
            )

        return output

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _slu_collate_fn(batch, pad_id=self.manifest_processor.pad_id)


def _slu_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, text_lengths, _, semantics_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, text_lengths, _, semantics_lengths, = packed_batch
    else:
        raise ValueError("Expects 6 or 7 tensors in the batch!")

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_text_len = max(text_lengths).item()
    max_semantics_len = max(semantics_lengths).item()

    audio_signal, texts, semantics = [], [], []
    for b in batch:
        if len(b) == 7:
            sig, sig_len, text_i, text_i_len, semantics_i, semantics_i_len, _ = b
        else:
            sig, sig_len, text_i, text_i_len, semantics_i, semantics_i_len = b

        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

        text_i_len = text_i_len.item()
        if text_i_len < max_text_len:
            pad = (0, max_text_len - text_i_len)
            text_i = torch.nn.functional.pad(text_i, pad, value=pad_id)
        texts.append(text_i)

        semantics_i_len = semantics_i_len.item()
        if semantics_i_len < max_semantics_len:
            pad = (0, max_semantics_len - semantics_i_len)
            semantics_i = torch.nn.functional.pad(semantics_i, pad, value=pad_id)
        semantics.append(semantics_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    texts = torch.stack(texts)
    text_lengths = torch.stack(text_lengths)

    semantics = torch.stack(semantics)
    semantics_lengths = torch.stack(semantics_lengths)

    if sample_ids is None:
        return audio_signal, audio_lengths, texts, text_lengths, semantics, semantics_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, texts, text_lengths, semantics, semantics_lengths, sample_ids


class ManifestProcessorSLU:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, durations (in seconds) and semantics.
    Each new line is a different sample. Example below:
    {
     "audio_filepath": "/path/to/audio.wav", "text": "set an alarm for ten o'clock",
     "semantics": [{"senario": "alarm", "action": "set", "entities": [{"slot": "time", "filler": "ten o'clock"}]}],
     "offset": 301.75, "duration": 0.82
    }
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        text_parser: Str for a language specific preprocessor or a callable.
        semantic_parser: Str for a semantic preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        text_parser: Union[str, Callable],
        semantic_parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):

        self.text_parser = text_parser
        self.semantic_parser = semantic_parser

        self.collection = collections.AudioTextSemantics(
            manifests_files=manifest_filepath,
            text_parser=text_parser,
            semantic_parser=semantic_parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, sample: collections.AudioTextSemantics.OUTPUT_TYPE) -> Tuple[List[int], int]:
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl

    def process_semantics_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_semantics_by_sample(sample)

    def process_semantics_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_semantics_by_sample(sample)

    def process_semantics_by_sample(self, sample: collections.AudioTextSemantics.OUTPUT_TYPE) -> Tuple[List[int], int]:
        t, tl = sample.semantic_tokens, len(sample.semantic_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


class AudioTextSemanticsBPEDataset(_AudioTextSemanticsDataset):
    def __init__(
        self,
        manifest_filepath: str,
        text_tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        semantic_tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(semantic_tokenizer, 'bos_token'):
            bos_id = semantic_tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(semantic_tokenizer, 'eos_token'):
            eos_id = semantic_tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(semantic_tokenizer, 'pad_token'):
            pad_id = semantic_tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            text_parser=TokenizerWrapper(text_tokenizer),
            semantic_parser=TokenizerWrapper(semantic_tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
        )


class EnCharTokenizer(tokenizers.TokenizerSpec):
    """
    Creates a basic char tokenizer for text given a tokenizer for semantics
    """

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.vocab = [
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

        self.special_token_to_prepend = None
        self.special_token_to_append = None
        self.special_token_ids_to_remove_in_decoding = []
        special_tokens = ["mask_token", "bos_token", "eos_token", "pad_token", "sep_token", "cls_token", "unk_token"]
        for sp_token in special_tokens:
            if hasattr(tokenizer, sp_token):
                idx = getattr(tokenizer, sp_token.replace("_token", "_id"))
                self.vocab.insert(idx, sp_token)
                if sp_token != "unk_token":
                    self.special_token_ids_to_remove_in_decoding.append(idx)
                if sp_token == "bos_token":
                    self.special_token_to_prepend = "<s>"
                elif sp_token == "eos_token":
                    self.special_token_to_append = "</s>"

        self.inv_vocab = dict()
        for idx, char in enumerate(self.vocab):
            self.inv_vocab[char] = idx

    def text_to_tokens(self, text: str) -> List[str]:
        token_candidates = [char for char in text]
        tokens = []
        if self.special_token_to_prepend is not None:
            tokens.append(getattr(self, self.special_token_to_prepend))
        for i, token in enumerate(token_candidates):
            if token in self.vocab:
                tokens.append(token)
            elif self.unk_token is not None:
                tokens.append(self.unk_token)
            else:
                warnings.warn(
                    f"Character {repr(token)} in position {i} is not present in vocabulary and no `<UNK>` token was "
                    f"set. Character {repr(token)} is discarded."
                )
        if self.special_token_to_append is not None:
            tokens.append(self.special_token_to_append)
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab[id] for id in ids]

    def ids_to_text(self, ids: List[int]) -> str:
        ids_ = [id_ for id_ in ids if id_ not in self.special_token_ids_to_remove_in_decoding]
        return "".join(self.ids_to_tokens(ids_))


def get_slu_bpe_dataset(
    config: dict,
    text_tokenizer: 'TokenizerSpec',
    semantic_tokenizer: 'TokenizerSpec',
    augmentor: Optional['AudioAugmentor'] = None,
) -> AudioTextSemanticsBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = AudioTextSemanticsBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        text_tokenizer=text_tokenizer,
        semantic_tokenizer=semantic_tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        return_sample_id=config.get('return_sample_id', False),
    )
    return dataset
