# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn

from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron as MLP
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging


class ConcatPooling(nn.Module):
    def __init__(self, pooling_factor):
        super().__init__()
        self.pooling_factor = pooling_factor

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        if seq_len % self.pooling_factor != 0:
            x = x[:, : -(seq_len % self.pooling_factor), :]
        x = x.view(batch_size, seq_len // self.pooling_factor, input_dim * self.pooling_factor)
        return x


class PoolingMLPConnectors(NeuralModule, Exportable, AccessMixin):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        num_layers: int = 2,
        activation: str = "relu",
        pooling: str = "mean",
        pooling_factor: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.num_layers = num_layers
        self.activation = activation
        self.pooling = pooling
        self.pooling_factor = pooling_factor

        if pooling == "cat":
            self.preprocess = nn.Sequential(
                ConcatPooling(pooling_factor), nn.Linear(input_dim * pooling_factor, hidden_dim)
            )
        else:
            self.preprocess = nn.Sequential(
                nn.AvgPool1d(pooling_factor, stride=pooling_factor), nn.Linear(input_dim, hidden_dim)
            )

        self.mlp = MLP(hidden_dim, output_dim, num_layers, activation, log_softmax=False)

    def forward(self, audio_signal, audio_signal_len=None):
        """
        Args: 
            audio_signal: [batch_size, seq_len, input_dim]
            audio_signal_len: [batch_size]
        Returns:
            outputs: [batch_size, seq_len/pooling_factor, output_dim]
            outputs_len: [batch_size]
        """
        outputs = self.preprocess(audio_signal)
        outputs = self.mlp(outputs)
        outputs_len = torch.div(audio_signal_len, self.pooling_factor, rounding_mode='floor')
        return outputs, outputs_len
