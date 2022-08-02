import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from slurp_eval_tools.metrics import ErrorMetric
from tqdm.auto import tqdm

from .slu_utils import SearcherConfig, parse_semantics_str2dict
from nemo.collections.asr.models import ASRModel
from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


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


@dataclass
class InferenceConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    mode: str = "oracle"
    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 8

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    searcher: SearcherConfig = SearcherConfig(type="greedy")


def load_manifest(filepath):
    data = []
    with open(filepath, "r") as fin:
        for line in fin.readlines():
            datum = json.loads(line)
            data.append(datum)
    return data


def get_query_data(manifest: List[dict], key: str):
    results = []
    for datum in manifest:
        results.append(datum[key])
    return results


def parse_semantics_dict(queries: List[str], intents: List[str], slots: List[str]) -> List[Dict]:
    results = []
    for query_i, intent_i, slots_i in zip(queries, intents, slots):
        datum = {}
        scenario = intent_i.split("_")[0]
        action = intent_i.strip(scenario + "_")
        datum["scenario"] = scenario
        datum["action"] = action

        query_i = query_i.split()
        slots_i = slots_i.split()
        entities = []
        curr_slot = None
        curr_filler = []
        for j in range(len(query_i)):
            if slots_i[j].startswith("B-"):
                if curr_slot is not None:
                    entity = {"type": curr_slot, "filler": " ".join(curr_filler)}
                    entities.append(entity)
                curr_slot = slots_i[j].strip("B-")
                curr_filler = [query_i[j]]
            elif slots_i[j].startswith("I-") and slots_i[j].strip("I-") == curr_slot:
                curr_filler.append(query_i[j])
            elif curr_slot is not None:
                entity = {"type": curr_slot, "filler": " ".join(curr_filler)}
                entities.append(entity)
                curr_slot = None
                curr_filler = []

        if curr_slot is not None:
            entity = {"type": curr_slot, "filler": " ".join(curr_filler)}
            entities.append(entity)
        datum["entities"] = entities
        results.append(datum)
    return results


def slurp_inference(
    nlu_model, path2manifest: str, batch_size: int = 4, num_workers: int = 0, mode: str = "oracle"
) -> List[str]:

    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # Model's mode and device
    is_training = nlu_model.training
    device = next(nlu_model.parameters()).device

    test_manifest = load_manifest(path2manifest)

    try:
        # Switch model to evaluation mode
        nlu_model.eval()

        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle': False,
            'pin_memory': True,
            'drop_last': False,
        }

        config = DictConfig(config)

        if "oracle" in mode:
            print("------- using oracle text -------")
            key = "text"
        else:
            print("------ using predicted text ------")
            key = "pred_text"

        print("loading manifest...")
        queries = get_query_data(test_manifest, key)

        print("predicting...")
        pred_intents, pred_slots = nlu_model.predict_from_examples(queries, config)

        print("parsing output to semantics dict...")
        hypotheses = parse_semantics_dict(queries, pred_intents, pred_slots)

    finally:
        # set mode back to its original value
        nlu_model.train(mode=is_training)
        logging.set_verbosity(logging_level)
    return hypotheses, pred_intents, pred_slots, queries


@hydra_runner(config_name="InferenceConfig", schema=InferenceConfig)
def main(cfg: InferenceConfig) -> InferenceConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        logging.info(f"Restoring model : {cfg.model_path}")
        nlu_model = IntentSlotClassificationModel.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        nlu_model = IntentSlotClassificationModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    nlu_model.set_trainer(trainer)
    nlu_model = nlu_model.eval()

    # get audio filenames
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        # get filenames from manifest
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
            return None

        manifest_dir = Path(cfg.dataset_manifest).parent
        with open(cfg.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                audio_file = Path(item['audio_filepath'])
                if not audio_file.is_file() and not audio_file.is_absolute():
                    audio_file = manifest_dir / audio_file
                filepaths.append(str(audio_file.absolute()))

    logging.info(f"\nStart inference with {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )

        return cfg

    # transcribe audio
    with autocast():
        with torch.no_grad():
            predictions, pred_intents, pred_slots, queries = slurp_inference(
                nlu_model=nlu_model,
                path2manifest=cfg.dataset_manifest,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                mode=cfg.mode,
            )

    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # write audio transcriptions
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.audio_dir is not None:
            for idx, text in enumerate(predictions):
                item = {'audio_filepath': filepaths[idx], 'pred_semantics': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['pred_semantics'] = predictions[idx]
                    item['pred_intent'] = pred_intents[idx]
                    item['pred_slots'] = pred_slots[idx]
                    item['query'] = queries[idx]
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing predictions !")
    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
