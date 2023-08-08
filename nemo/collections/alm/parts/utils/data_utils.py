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

import numpy as np
import torch


def maybe_cast_to_list(x):
    if isinstance(x, np.ndarray):
        return [item.tolist() for item in x]
    return x


def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m


@torch.no_grad()
def create_attention_mask(max_length):
    """Create `attention_mask`.
    Args:
        input_ids: A 1D tensor that holds the indices of tokens.
    """
    # seq_length = len(input_ids)
    # `attention_mask` has the shape of [1, seq_length, seq_length]
    attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
    attention_mask = attention_mask < 0.5
    return attention_mask


def get_num_samples_from_files(file_list):
    if isinstance(file_list, str):
        file_list = file_list.split(',')
    num_samples = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = list(f.readlines())
            num = len(lines)
            if lines[-1] == '\n':
                num -= 1
            num_samples.append(num)
    return num_samples
