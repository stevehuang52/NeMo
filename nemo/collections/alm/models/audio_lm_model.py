
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
import torch
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.alm.data.audio_text_qa_dataset import AudioQuestionAnswerDataset
from nemo.collections.alm.parts.utils.data_utils import create_attention_mask
from nemo.collections.asr.models import ASRModel
from nemo.collections.nlp.models.language_modeling.megatron_llama_model import MegatronLLAMAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTLoRAModel
from nemo.collections.nlp.models.language_modeling.llama.llama_model import LLAMAModel
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging

try:
    import apex.transformer.pipeline_parallel.utils
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class AudioGPTLoRAModel(MegatronGPTLoRAModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer)
        self._setup_audio_encoder(cfg.audio_encoder)
        self.connector = self._setup_connector(cfg.connector)
        self.configure_optimizers()

    def _setup_audio_encoder(self, cfg: DictConfig):
        self.audio_preprocessor = ASRModel.from_config_dict(cfg.preprocessor)
        if hasattr(cfg, 'spec_augment') and cfg.spec_augment is not None:
            self.audio_spec_augmentation = ASRModel.from_config_dict(cfg.spec_augment)
        else:
            self.audio_spec_augmentation = None
        self.audio_encoder = ASRModel.from_config_dict(cfg.encoder)

    def _setup_connector(self, cfg: DictConfig):
        if 'output_dim' not in cfg:
            with open_dict(cfg):
                cfg.output_dim = self.model.language_model.embedding.word_embeddings.weight.shape[1]
        return ASRModel.from_config_dict(cfg)

    def forward_audio_encoder(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
        Returns:
            A tuple of 2 elements -
            1) The audio feature tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.audio_preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.audio_spec_augmentation is not None and self.training:
            processed_signal = self.audio_spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.audio_encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0].transpose(1, 2)  # [B, D, T] -> [B, T, D]
        encoded_len = encoder_output[1]
        audio_feat, audio_feat_len = self.connector(encoded, encoded_len)
        return audio_feat, audio_feat_len

    def _get_loss_mask(self, batch, audio_feats, audio_feat_lens=None):
        return torch.cat([torch.zeros(audio_feats.size(0), audio_feats.size(1), device=audio_feats.device, dtype=batch['loss_mask'].dtype), batch['loss_mask']], dim=1)
    
    def _get_padded_labels(self, batch, audio_feats, audio_feat_lens=None):
        return torch.cat([torch.zeros(audio_feats.size(0), audio_feats.size(1), device=audio_feats.device, dtype=batch['labels'].dtype), batch['labels']], dim=1)

    def _get_attention_mask(self, batch, audio_feats, audio_feat_lens=None):
        """
        Args:
            batch: dict of tensors
            audio_feats: (batch_size, time, hidden_size)
            audio_feat_lens: (batch_size)
        Returns:
            mask: (batch_size, 1, total_seq_len, total_seq_len)
        """
        mask = create_attention_mask(audio_feats.size(1) + batch['tokens'].size(1)).to(audio_feats.device)
        return mask.repeat(audio_feats.size(0), 1, 1, 1)

    def _get_input_embeddings(self, audio_feats, tokens, tokens_position_ids):
        # Get the input embeddings to the LM
        # audio_feats: (batch_size, time, hidden_size)
        # tokens: (batch_size, seq_len)
        # tokens_position_ids: (batch_size, seq_len)
        # return: (seq_len, batch_size, hidden_size)
        lm_embedding = self.model.language_model.embedding
        text_embeddings = lm_embedding.word_embeddings(tokens)  # (batch_size, seq_len, hidden_size)
        input_embeddings = torch.cat([audio_feats, text_embeddings], dim=1)

        audio_position_ids = torch.arange(audio_feats.size(1), device=audio_feats.device).repeat(audio_feats.size(0), 1)
        tokens_position_ids = tokens_position_ids + audio_feats.size(1)
        position_ids = torch.cat([audio_position_ids, tokens_position_ids], dim=1)

        if hasattr(lm_embedding, 'position_embeddings'):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            input_embeddings = input_embeddings + position_embeddings
        
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if lm_embedding.transpose_batch_sequence:
            input_embeddings = input_embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if lm_embedding.fp32_residual_connection:
            input_embeddings = input_embeddings.float()

        # Dropout.
        if lm_embedding.sequence_parallel:
            input_embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(input_embeddings)
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                input_embeddings = lm_embedding.embedding_dropout(input_embeddings)
        else:
            input_embeddings = lm_embedding.embedding_dropout(input_embeddings)
        
        return input_embeddings

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = next(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids', 'audio_signal', 'audio_signal_length'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            audio_feats, audio_feat_lens = self.forward_audio_encoder(
                input_signal=batch['audio_signal'],
                input_signal_length=batch['audio_signal_length']
            )

            loss_mask = self._get_loss_mask(batch, audio_feats, audio_feat_lens)
            attention_mask = self._get_attention_mask(batch, audio_feats, audio_feat_lens)
            encoder_input = self._get_input_embeddings(audio_feats, batch['tokens'], batch['position_ids'])
            labels = self._get_padded_labels(batch, audio_feats, audio_feat_lens)

            # Model forward pass
            output_tensor = model(
                input_ids=None,
                position_ids=None,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )

            output_tensor = output_tensor[0]  # get loss only, ingore logits

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(loss_mask, output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def _build_dataset(self, data_cfg, is_train=True):
        datasets = []

        if isinstance(data_cfg.file_names, str):
            file_names = data_cfg.file_names.split(',')
        else:
            file_names = data_cfg.file_names

        if is_train and not data_cfg.get('is_tarred', False):
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            concat_sampling_probabilities = data_cfg.get('concat_sampling_probabilities', None)
            if concat_sampling_probabilities is None:
                concat_sampling_probabilities = [1.0 / len(file_names)] * len(file_names)
            elif len(data_cfg.get('concat_sampling_probabilities', None)) != len(file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(concat_sampling_probabilities, file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        if 'augmentor' in data_cfg:
            augmentor = process_augmentations(data_cfg['augmentor'], global_rank=self.global_rank, world_size=self.world_size)
        else:
            augmentor = None
        for file_path, num_samples in zip(file_names, num_train_samples_per_dataset):
            dataset = AudioQuestionAnswerDataset(
                manifest_filepath=file_path,
                tokenizer=self.tokenizer,
                sample_rate=data_cfg.sample_rate,
                int_values=data_cfg.get('int_values', False),
                augmentor=augmentor,
                max_duration=getattr(data_cfg, 'max_duration', None),
                min_duration=getattr(data_cfg, 'min_duration', None),
                max_utts=getattr(data_cfg, 'max_utts', -1),
                trim=getattr(data_cfg, 'trim_silence', False),
                channel_selector=getattr(data_cfg, 'channel_selector', None),
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=False,
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
            )
            datasets.append(dataset)

        if is_train and not data_cfg.get('is_tarred', False):
            dataset = BlendableDataset(
                datasets=datasets, weights=concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets