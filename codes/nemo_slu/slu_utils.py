import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import ipdb
import torch
from omegaconf import DictConfig
from slurp_eval_tools.metrics import ErrorMetric

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.transformer import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
    TopKSequenceGenerator,
)
from nemo.core.classes.module import NeuralModule


@dataclass
class SearcherConfig:
    type: str = "greedy"  # choices=[greedy, topk, beam]
    max_sequence_length: int = 512
    max_delta_length: int = -1
    temperature: float = 1.0  # for top-k sampling
    beam_size: int = 1  # K for top-k sampling, N for beam search
    len_pen: float = 0.0  # for beam-search
    teacher_force_greedy: bool = False


class SequenceGenerator:
    TYPE_GREEDY = "greedy"
    TYPE_TOPK = "topk"
    TYPE_BEAM = "beam"
    SEARCHER_TYPES = [TYPE_GREEDY, TYPE_TOPK, TYPE_BEAM]

    def __init__(
        self,
        cfg: DictConfig,
        embedding: NeuralModule,
        decoder: NeuralModule,
        log_softmax: NeuralModule,
        tokenizer: TokenizerSpec,
    ) -> None:
        super().__init__()

        self._type = cfg.get("type", "greedy")
        self.tokenizer = tokenizer
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self.eos_id = getattr(tokenizer, "eos_id", -1)
        self.bos_id = getattr(tokenizer, "bos_id", -1)
        common_args = {
            "pad": self.pad_id,
            "bos": self.bos_id,
            "eos": self.eos_id,
            "max_sequence_length": cfg.get("max_sequence_length", 512),
            "max_delta_length": cfg.get("max_delta_length", -1),
            "batch_size": cfg.get("batch_size", 1),
        }
        if self._type == self.TYPE_GREEDY:
            self.generator = GreedySequenceGenerator(embedding, decoder, log_softmax, **common_args)
        elif self._type == self.TYPE_TOPK:
            beam_size = cfg.get("beam_size", 1)
            temperature = cfg.get("temperature", 1.0)
            self.generator = TopKSequenceGenerator(
                embedding, decoder, log_softmax, beam_size, temperature, **common_args
            )
        elif self._type == self.TYPE_BEAM:
            beam_size = cfg.get("beam_size", 1)
            len_pen = cfg.get("len_pen", 0.0)
            self.generator = BeamSearchSequenceGenerator(
                embedding, decoder, log_softmax, beam_size, len_pen, **common_args
            )
        else:
            raise ValueError(
                f"Sequence Generator only supports one of {self.SEARCH_TYPES}, but got {self._type} instead."
            )

    def __call__(
        self,
        encoder_states,
        encoder_input_mask=None,
        return_beam_scores=False,
        pad_max_len: Optional[int] = None,
        **kwargs,
    ):
        predictions = self.generator(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_input_mask,
            return_beam_scores=return_beam_scores,
        )

        if pad_max_len:
            predictions = pad_sequence(predictions, pad_max_len, self.pad_id)
        return predictions

    def get_seq_length(self, seq):
        lengths = seq.size(1) * torch.ones(seq.size(0), device=seq.device).long()
        pos = (seq == self.eos_id).long().nonzero()
        seq_lengths = torch.scatter(lengths, dim=0, index=pos[:, 0], src=pos[:, 1])
        return seq_lengths

    def decode_semantics_from_tokens(self, seq_tokens):
        semantics_list = []
        # Drop sequence tokens to CPU
        seq_tokens = seq_tokens.detach().long().cpu()
        seq_lengths = self.get_seq_length(seq_tokens)
        # iterate over batch
        for ind in range(seq_tokens.shape[0]):
            tokens = seq_tokens[ind].numpy().tolist()
            length = seq_lengths[ind].long().cpu().item()
            tokens = tokens[:length]
            text = "".join(self.tokenizer.tokenizer.decode_ids(tokens))
            semantics_list.append(text)
        return semantics_list


def pad_sequence(seq: torch.Tensor, max_len: int, pad_token: int = 0) -> torch.Tensor:
    """
    params:
        - seq: integer token sequences of shape BxT
        - max_len: integer for max sequence length
        - pad_token: integer token for padding
    return:
        - padded sequence of shape B x max_len
    """
    batch = seq.size(0)
    curr_len = seq.size(1)
    if curr_len >= max_len:
        return seq

    padding = torch.zeros(batch, max_len - curr_len, dtype=seq.dtype, device=seq.device).fill_(pad_token)
    return torch.cat([seq, padding], dim=1)


def parse_semantics_str2dict(semantics_str: Union[List[str], str]) -> Dict:
    invalid = False
    if isinstance(semantics_str, list):
        semantics_str = " ".join(semantics_str)
    try:
        _dict = ast.literal_eval(semantics_str.replace("|", ","))
        if not isinstance(_dict, dict):
            _dict = {
                "scenario": "none",
                "action": "none",
                "entities": [],
            }
            invalid = True
    except SyntaxError:  # need this if the output is not a valid dictionary
        _dict = {
            "scenario": "none",
            "action": "none",
            "entities": [],
        }
        invalid = True

    if not isinstance(_dict["scenario"], str):
        _dict["scenario"] = "none"
        invalid = True
    if not isinstance(_dict["action"], str):
        _dict["action"] = "none"
        invalid = True
    if "entities" not in _dict:
        _dict["entities"] = []
        invalid = True
    else:
        for i, x in enumerate(_dict["entities"]):
            item, entity_error = parse_entity(x)
            invalid = invalid or entity_error
            _dict["entities"][i] = item

    return _dict, invalid


def parse_entity(item: Dict):
    error = False
    for key in ["type", "filler"]:
        if key not in item or not isinstance(item[key], str):
            item[key] = "none"
            error = True
    return item, error


def get_seq_mask(seq: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    get the sequence mask based on the actual length of each sequence
    input:
        - seq: tensor of shape [BxLxD]
        - seq_len: tensor of shape [B]
    output:
        - binary mask of shape [BxL]
    """
    mask = torch.arange(seq.size(1))[None, :].to(seq.device) < seq_lens[:, None]
    return mask.to(seq.device, dtype=bool)


class SLUEvaluator:
    def __init__(self, average_mode: str = 'micro') -> None:
        if average_mode not in ['micro', 'macro']:
            raise ValueError(f"Only supports 'micro' or 'macro' average, but got {average_mode} instead.")
        self.average_mode = average_mode
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=average_mode)
        self.invalid = 0
        self.total = 0

    def reset(self):
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=self.average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=self.average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=self.average_mode)
        self.invalid = 0
        self.total = 0

    def update(self, predictions: Union[List[str], str], groundtruth: Union[List[str], str]) -> None:
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]

        for pred, truth in zip(predictions, groundtruth):
            pred, syntax_error = parse_semantics_str2dict(pred)
            truth, _ = parse_semantics_str2dict(truth)
            self.scenario_f1(truth["scenario"], pred["scenario"])
            self.action_f1(truth["action"], pred["action"])
            self.intent_f1(f"{truth['scenario']}_{truth['action']}", f"{pred['scenario']}_{pred['action']}")
            self.span_f1(truth["entities"], pred["entities"])
            for distance, metric in self.distance_metrics.items():
                metric(truth["entities"], pred["entities"])

            self.total += 1
            self.invalid += int(syntax_error)

    def compute(self, aggregate=True) -> Dict:
        scenario_results = self.scenario_f1.get_metric()
        action_results = self.action_f1.get_metric()
        intent_results = self.intent_f1.get_metric()
        entity_results = self.span_f1.get_metric()
        word_dist_results = self.distance_metrics['word'].get_metric()
        char_dist_results = self.distance_metrics['char'].get_metric()
        self.slu_f1(word_dist_results)
        self.slu_f1(char_dist_results)
        slurp_results = self.slu_f1.get_metric()

        if not aggregate:
            return {
                "scenario": scenario_results,
                "action": action_results,
                "intent": intent_results,
                "entity": entity_results,
                "word_dist": word_dist_results,
                "char_dist": char_dist_results,
                "slurp": slurp_results,
                "invalid": self.invalid,
                "total": self.total,
            }

        scores = dict()
        scores["invalid"] = self.invalid
        scores["total"] = self.total
        self.update_scores_dict(scenario_results, scores, "scenario")
        self.update_scores_dict(action_results, scores, "action")
        self.update_scores_dict(intent_results, scores, "intent")
        self.update_scores_dict(entity_results, scores, "entity")
        self.update_scores_dict(word_dist_results, scores, "word_dist")
        self.update_scores_dict(char_dist_results, scores, "char_dist")
        self.update_scores_dict(slurp_results, scores, "slurp")

        return scores

    def update_scores_dict(self, source: Dict, target: Dict, tag: str = '') -> Dict:
        scores = source['overall']
        p, r, f1 = scores[:3]
        target[f"{tag}_p"] = p
        target[f"{tag}_r"] = r
        target[f"{tag}_f1"] = f1
        return target


def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name, own_state[name].size(), param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch
