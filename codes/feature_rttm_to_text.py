import contextlib
import json
import math
import os
import random
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from jiwer import wer
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.data import feature_to_text_dataset
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    use_rttm: bool = True
    normalize: Optional[str] = "post_norm"  # choices=[pre_norm, post_norm]
    frame_unit_time_secs: float = 0.01  # unit time per frame in seconds

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 8
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.dataset_manifest is None:
        raise ValueError("cfg.dataset_manifest cannot be None!")

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
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    asr_model = asr_model.eval()

    # set True to collect additional transcription information
    return_hypotheses = False

    # Setup decoding strategy
    is_rnnt = False
    if hasattr(asr_model, 'change_decoding_strategy'):
        # Check if ctc or rnnt model
        if hasattr(asr_model, 'joint'):  # RNNT model
            is_rnnt = True
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
            decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)
            decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.pred_name_postfix is not None:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{cfg.pred_name_postfix}.json')
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    # Setup dataloader
    data_config = {
        "manifest_filepath": cfg.dataset_manifest,
        "normalize": cfg.normalize,
        "frame_unit_time_secs": cfg.frame_unit_time_secs,
        "use_rttm": cfg.use_rttm,
    }
    logging.info(f"use_rttm={cfg.use_rttm}")
    if hasattr(asr_model, "tokenizer"):
        dataset = feature_to_text_dataset.get_bpe_dataset(config=data_config, tokenizer=asr_model.tokenizer)
    else:
        data_config["labels"] = asr_model.decoder.vocabulary
        dataset = feature_to_text_dataset.get_char_dataset(config=data_config)
    logging.info(f"Transcribing {len(dataset)} files...")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        collate_fn=dataset._collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    hypotheses = []
    all_hypotheses = []
    with autocast():
        with torch.no_grad():
            for test_batch in tqdm(dataloader, desc="Transcribing"):
                outputs = asr_model.forward(
                    processed_signal=test_batch[0].to(map_location),
                    processed_signal_length=test_batch[1].to(map_location),
                )

                logits, logits_len = outputs[0], outputs[1]

                current_hypotheses, all_hyp = decode_function(logits, logits_len, return_hypotheses=return_hypotheses,)

                if return_hypotheses and not is_rnnt:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                        if current_hypotheses[idx].alignments is None:
                            current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                hypotheses += current_hypotheses
                if all_hyp is not None:
                    all_hypotheses += all_hyp
                else:
                    all_hypotheses += current_hypotheses

                del logits
                del test_batch

    # Save output to manifest
    manifest_data = load_manifest(cfg.dataset_manifest)
    groundtruth = []
    for i in range(len(manifest_data)):
        manifest_data[i]["pred_text"] = hypotheses[i]
        groundtruth.append(manifest_data[i]["text"])
    save_manifest(manifest_data, cfg.output_filename)
    logging.info(f"Output saved at {cfg.output_filename}")
    wer_score = wer(truth=groundtruth, hypothesis=hypotheses)
    logging.info("-----------------------------------------")
    logging.info(f"WER={wer_score:.4f}")
    logging.info("-----------------------------------------")


def save_manifest(manifest_data, out_file):
    with Path(out_file).open("w") as fout:
        for item in manifest_data:
            fout.write(f"{json.dumps(item)}\n")


def load_manifest(manifest_file):
    data = []
    with Path(manifest_file).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    main()
