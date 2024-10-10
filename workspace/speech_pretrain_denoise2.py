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


import os
import time

import pytorch_lightning as pl
from omegaconf import OmegaConf
from one_logger_utils_nemo import OneLoggerNemoModel, OneLoggerPTLTrainer

from nemo.collections.asr.models.ssl_models_v2 import EncDecSpeechDenoiseMLMModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
# Example of unsupervised pre-training of a model
```sh
python speech_pre_training.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp"  \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Namex of project>"
```

For documentation on fine-tuning, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations
When doing supervised fine-tuning from unsupervised pre-trained encoder, set flag init_strict to False

"""


@hydra_runner(config_path="./configs", config_name="conformer_large_ssl_rq")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    model_name = str(cfg.exp_manager.name).split("_")[1]
    # only include precision and num nodes
    suffix = str(cfg.exp_manager.name).split("_Pc")[1].split("_r")[0]
    one_logger_callback_config = {
        "enable_for_current_rank": os.environ.get('RANK') == '0',
        "one_logger_async": cfg.get("exp_manager").get("create_wandb_logger", False),
        "log_every_n_train_iterations": cfg.get("trainer").get("log_every_n_steps", 10),
        "app_tag_run_version": "0.0.0",
        "summary_data_schema_version": "1.0.0",
        "app_run_type": "training",
        "app_tag": cfg.exp_manager.name,  # Please change this
        "app_tag_run_name": f"{model_name}-{suffix}",  # Please change this
        "one_logger_project": "heh-NEST-train",  # Please change this
        "one_logger_run_name": cfg.exp_manager.name,  # Please change this
        "world_size": os.environ.get('WORLD_SIZE', -1),
        "global_batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
        "batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
        "train_iterations_target": cfg.get("trainer").get("max_steps", 1),
        "train_samples_target": cfg.get("trainer").get("max_steps", 1)
        * cfg.get("model").get("train_ds").get("batch_size", 1),
        "is_train_iterations_enabled": True,
        "is_baseline_run": False,
        "is_test_iterations_enabled": False,
        "is_validation_iterations_enabled": True,
        "is_save_checkpoint_enabled": True,
        "is_log_throughput_enabled": False,
        "micro_batch_size": cfg.get("model").get("train_ds").get("batch_size", 1),
        "seq_length": 1,
        "save_checkpoint_strategy": "sync",
    }

    # trainer = pl.Trainer(**cfg.trainer)

    trainer = OneLoggerPTLTrainer(trainer_config=cfg.trainer, callback_config=one_logger_callback_config)
    # trainer.callbacks.append(OneLoggerPTLCallback())
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = OneLoggerNemoModel(model_class=EncDecSpeechDenoiseMLMModel, cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
