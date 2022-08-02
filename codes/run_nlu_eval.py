import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import nemo_slu.eval_utils_nlu2 as inference
import torch
from nemo_slu.eval_utils_nlu import InferenceConfig, SLUEvaluator
from omegaconf import MISSING, OmegaConf, open_dict
from slurp_eval_tools.util import format_results

from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class EvaluationConfig(InferenceConfig):
    dataset_manifest: str = MISSING
    output_filename: Optional[str] = "evaluation_transcripts.json"
    average: str = "micro"
    full: bool = False
    errors: bool = False
    table_layout: str = "fancy_grid"
    only_score_manifest: bool = False
    mode: str = "nlu"


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    torch.set_grad_enabled(False)

    cfg.output_filename = str(Path(Path(cfg.model_path).parent) / Path("predictions.json"))

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.only_score_manifest:
        # Transcribe speech into an output directory
        transcription_cfg = inference.main(cfg)  # type: EvaluationConfig

        # Release GPU memory if it was used during transcription
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("Finished transcribing speech dataset. Computing metrics..")

    else:
        cfg.output_filename = cfg.dataset_manifest
        transcription_cfg = cfg

    ground_truth_text = []
    predicted_text = []
    invalid_manifest = False

    with open(transcription_cfg.output_filename, 'r') as f:
        for line in f:
            data = json.loads(line)

            if 'pred_semantics' not in data:
                invalid_manifest = True
                break

            ground_truth_text.append(data['semantics'])
            predicted_text.append(data['pred_semantics'])

    # Test for invalid manifest supplied
    if invalid_manifest:
        raise ValueError(
            f"Invalid manifest provided: {transcription_cfg.output_filename} does not "
            f"contain value for `pred_semantics`."
        )

    # Compute the metrics
    evaluator = SLUEvaluator(cfg.average)
    evaluator.update(predictions=predicted_text, groundtruth=ground_truth_text)
    results = evaluator.compute(aggregate=False)
    total = results["total"]
    invalid = results["invalid"]
    slurp_f1 = results["slurp"]["overall"][2]

    print("-------------- Results --------------")
    print(
        format_results(
            results=results["scenario"],
            label="scenario",
            full=cfg.full,
            errors=cfg.errors,
            table_layout=cfg.table_layout,
        ),
        "\n",
    )

    print(
        format_results(
            results=results["action"], label="action", full=cfg.full, errors=cfg.errors, table_layout=cfg.table_layout
        ),
        "\n",
    )

    print(
        format_results(
            results=results["intent"],
            label="intent (scen_act)",
            full=cfg.full,
            errors=cfg.errors,
            table_layout=cfg.table_layout,
        ),
        "\n",
    )

    print(
        format_results(
            results=results["entity"],
            label="entities",
            full=cfg.full,
            errors=cfg.errors,
            table_layout=cfg.table_layout,
        ),
        "\n",
    )

    print(
        format_results(
            results=results["word_dist"],
            label="entities (word distance)",
            full=cfg.full,
            errors=cfg.errors,
            table_layout=cfg.table_layout,
        ),
        "\n",
    )

    print(
        format_results(
            results=results["char_dist"],
            label="entities (char distance)",
            full=cfg.full,
            errors=cfg.errors,
            table_layout=cfg.table_layout,
        ),
        "\n",
    )

    print(
        format_results(
            results=results["slurp"], label="SLU F1", full=cfg.full, errors=cfg.errors, table_layout=cfg.table_layout
        ),
        "\n",
    )

    print(f"Found {invalid} out of {total} predictions that have syntax error.")

    # Inject the metric name and score into the config, and return the entire config
    with open_dict(cfg):
        cfg.metric_name = "slurp_f1"
        cfg.metric_value = slurp_f1

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
