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

import copy
import json
import os
import tempfile
from abc import abstractmethod
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging, model_utils

from src.audio_to_multi_label import get_audio_multi_label_dataset


class EncDecMultiClassificationModel(EncDecClassificationModel):

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'C', 'T'), LogitsType())}

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        return []

    def _setup_loss(self):
        return CrossEntropyLoss(logits_ndim=3)

    def _setup_dataloader_from_config(self, config: DictConfig):
        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
            return None

        dataset = get_audio_multi_label_dataset(config)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config.get('shuffle', False),
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def get_label_masks(self, labels, labels_len):
        mask = torch.arange(labels.size(1))[None, :].to(labels.device) < labels_len[:, None]
        return mask.to(labels.device, dtype=bool)

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits = self.decoder(encoded.transpose(1,2))
        return logits

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, labels_len)
        masks = self.get_label_masks(labels, labels_len)

        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        self._accuracy(logits=logits.view(-1, logits.size(-1)), labels=labels.view(-1))
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('training_batch_accuracy_top@{}'.format(top_k), score)

        return {
            'loss': loss_value,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, labels_len)
        masks = self.get_label_masks(labels, labels_len)
        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)
        acc = self._accuracy(logits=logits.view(-1, logits.size(-1)), labels=labels.view(-1))
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, labels_len)
        masks = self.get_label_masks(labels, labels_len)
        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)
        acc = self._accuracy(logits=logits.view(-1, logits.size(-1)), labels=labels.view(-1))
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        return {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc': acc,
        }

    def reshape_labels(self, logits, labels, labels_len):
        logits_max_len = logits.size(1)
        labels_max_len = labels.size(1)
        batch_size = logits.size(0)
        if logits_max_len < labels_max_len:
            ratio = labels_max_len // logits_max_len
            res = labels_max_len % logits_max_len
            if res > 0:
                labels = labels[:, :-res]
                mask = labels_len > (labels_max_len - res)
                labels_len = labels_len - mask * res
            labels = labels.view(batch_size, ratio, -1).amax(1)
            labels_len = torch.div(labels_len, ratio, rounding_mode="floor")
            return labels, labels_len
        elif logits_max_len > labels_max_len:
            ratio = ceil(logits_max_len / labels_max_len)
            labels = labels.repeat_interleave(ratio, dim=1)
            return self.reshape_labels(logits, labels, labels_len)
        else:
            return labels, labels_len



