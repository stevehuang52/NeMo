# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import abc
from typing import List, Tuple

import torch

import nemo.collections.nlp.modules.common.text_generation_strategy as text_generation_strategy
from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'

__all__ = ['AudioToTextGenerationStrategy']


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    boolean = boolean.unsqueeze(0).unsqueeze(-1)
    return (1 - boolean) * val1 + boolean * val2


class AudioToTextGenerationStrategy(text_generation_strategy.GPTModelTextGenerationStrategy):
    def __init__(self, model, audio_signal, audio_signal_length, **kwargs):
        super().__init__(model)

        audio_feats, audio_feat_lens = self.model.forward_audio_encoder(
            input_signal=audio_signal.to(self.model.device),
            input_signal_length=audio_signal_length.to(self.model.device),
        )
        self.audio_feats = audio_feats
        self.audio_feat_lens = audio_feat_lens
        self.audio_length_to_add = torch.tensor([audio_feats.shape[1]], device=self.model.device)

    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        micro_batch_size = context_tokens.shape[0]

        audio_feats = self.audio_feats
        audio_feat_lens = self.audio_feat_lens
        self.attention_mask = self.model._get_attention_mask({'tokens': context_tokens}, audio_feats, audio_feat_lens)

        text_position_ids = torch.arange(context_tokens.size(1), dtype=torch.long, device=self.model.device)
        text_position_ids = text_position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        audio_position_ids = torch.arange(audio_feats.size(1), dtype=torch.long, device=self.model.device).repeat(
            micro_batch_size, 1
        )
        self.position_ids = torch.cat([audio_position_ids, text_position_ids + audio_feats.size(1)], dim=1)

        input_embeddings = self.model._get_input_embeddings(
            audio_feats, audio_feat_lens, context_tokens, text_position_ids
        )
        new_context_tokens = torch.cat(
            [
                torch.zeros(
                    micro_batch_size, audio_feats.size(1), device=context_tokens.device, dtype=context_tokens.dtype
                ),
                context_tokens,
            ],
            dim=1,
        )

        return new_context_tokens, input_embeddings

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        context_lengths: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            embeddings2use = input_embeddings[:context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use)
            started = context_lengths <= context_length
            embeddings2use = switch(input_embeddings[context_length - 1].unsqueeze(0), embeddings2use, started)

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                if len(end_string) > 1:
                    continue
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]
                end_tokens.add(token_id)

            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.alm.models.audio_lm_model import AudioGPTLoRAModel

    if isinstance(model, AudioGPTLoRAModel):
        return AudioToTextGenerationStrategy(model, **args)
    else:
        return text_generation_strategy.model_inference_strategy_dispatcher(model, **args)
