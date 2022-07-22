import copy
from pathlib import Path
from typing import Dict, Optional

import torch
from genericpath import isfile
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from .slu_loss import SeqNLLLoss
from .slu_utils import SearcherConfig, SequenceGenerator, get_seq_mask
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.core import adapter_mixins
from nemo.core.classes import Serialization as ModuleBuilder
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging, model_utils

__all__ = ['SLU2ASREncDecBPEModel']


class SLU2ASREncDecBPEModel(EncDecCTCModelBPE):
    def __init__(self, cfg: DictConfig, trainer=None):
        if hasattr(cfg, "adapter") and getattr(cfg.adapter, "enabled", False):
            with open_dict(cfg):
                adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
                if adapter_metadata is not None:
                    cfg.encoder._target_ = adapter_metadata.adapter_class_path

        super().__init__(cfg=cfg, trainer=trainer)

        # Init encoder from SSL checkpoint
        logging.info(f"Loading pretrained encoder from: {self.cfg.ssl_pretrained.model}")
        if Path(self.cfg.ssl_pretrained.model).is_file():
            ssl_model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.restore_from(
                restore_path=self.cfg.ssl_pretrained.model
            )
        else:
            ssl_model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.from_pretrained(
                model_name=self.cfg.ssl_pretrained.model
            )
        self.encoder.load_state_dict(ssl_model.encoder.state_dict(), strict=False)
        del ssl_model

        if self.cfg.ssl_pretrained.freeze:
            logging.info("Freezing SSL encoder...")
            self.encoder.freeze()

        if hasattr(cfg, "adapter") and getattr(cfg.adapter, "enabled", False):
            logging.info("Using Adapters...")
            adapter_cfg = LinearAdapterConfig(
                in_features=self.cfg.encoder.d_model,  # conformer specific model dim. Every layer emits this dim at its output.
                dim=cfg.adapter.adapter_dim,  # the bottleneck dimension of the adapter
                activation=cfg.adapter.adapter_activation,  # activation used in bottleneck block
                norm_position=cfg.adapter.adapter_norm_position,  # whether to use LayerNorm at the beginning or the end of the adapter
            )
            try:
                self.add_adapter(name=cfg.adapter.adapter_name, cfg=adapter_cfg)
            except ValueError:
                logging.warning(f"Adapter name {cfg.adapter.adapter_name} already exists, skipping.")
            self.set_enabled_adapters(name=cfg.adapter.adapter_name, enabled=True)
            self.encoder.freeze()
            self.unfreeze_enabled_adapters()

        self.vocabulary = self.tokenizer.tokenizer.get_vocab()
        vocab_size = len(self.vocabulary)

        # Create embedding layer
        self.cfg.embedding["vocab_size"] = vocab_size
        self.embedding = ModuleBuilder.from_config_dict(self.cfg.embedding)

        # Create decoder
        # self.decoder = ModuleBuilder.from_config_dict(self.cfg.decoder)

        # Create token classifier
        self.cfg.classifier["num_classes"] = vocab_size
        self.classifier = ModuleBuilder.from_config_dict(self.cfg.classifier)

        self.loss = SeqNLLLoss(label_smoothing=getattr(self.cfg, "loss.label_smoothing", 0.0))

        self.teacher_force_greedy = getattr(cfg.searcher, "teacher_force_greedy", False)
        self.searcher = SequenceGenerator(cfg.searcher, self.embedding, self.decoder, self.classifier, self.tokenizer)

        # Setup metric objects
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=False,  # use naive decoding
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_semantics": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "target_semantics_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType(), optional=True),
        }

    @classmethod
    def add_zero_tokens(cls, seq: torch.Tensor, pos="bos") -> torch.Tensor:
        """
        input:
        -   seq: token sequence of shape BxT
        -   pos: position of the zero tokens, either bos or eos
        output:
        -   token sequence of shape Bx(T+1)
        """
        if pos not in ["bos", "eos"]:
            raise ValueError(f"Invalid zero token type: {pos}, only supports ['bos', 'eos']")
        batch_size = seq.size(0)
        zero_tokens = torch.zeros(batch_size, 1).to(seq.device).long()

        if pos == "bos":
            return torch.cat([zero_tokens, seq], dim=1)
        else:
            return torch.cat([seq, zero_tokens], dim=1)

    def set_decoding_strategy(self, cfg: SearcherConfig):
        max_len = getattr(self.searcher, "generator.max_seq_length", cfg.max_sequence_length)
        max_delta = getattr(self.searcher, "generator.max_delta_length", cfg.max_delta_length)
        cfg.max_sequence_length = max_len
        cfg.max_delta_length = max_delta
        self.searcher = SequenceGenerator(cfg, self.embedding, self.decoder, self.classifier, self.tokenizer)

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        target_semantics=None,
        target_semantics_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            target_semantics: Tensor that represents a batch of semantic tokens,
                of shape [B, L].
            target_semantics_length: Vector of length B, that contains the individual lengths of the semantic
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
        encoded_mask = get_seq_mask(encoded, encoded_len)

        if target_semantics is None:
            predictions = self.searcher(encoded, encoded_mask)
            return None, None, predictions

        bos_semantics_tokens = target_semantics[:, :-1]
        bos_semantics = self.embedding(bos_semantics_tokens)
        bos_semantics_mask = get_seq_mask(bos_semantics, target_semantics_length - 1)

        decoded = self.decoder(
            encoder_states=encoded,
            encoder_mask=encoded_mask,
            decoder_states=bos_semantics,
            decoder_mask=bos_semantics_mask,
        )
        log_probs = self.classifier(decoded)

        if self.training or self.teacher_force_greedy:
            predictions = log_probs.argmax(dim=-1, keepdim=False)
        else:
            predictions = self.searcher(encoded, encoded_mask)

        pred_len = self.searcher.get_seq_length(predictions)
        return log_probs, pred_len, predictions

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        if len(batch) == 4:
            signal, signal_len, semantics, semantics_len = batch
        elif len(batch) == 5:
            signal, signal_len, semantics, semantics_len, idx = batch
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, semantics, semantics_len = batch
        else:
            signal, signal_len, transcript, transcript_len, semantics, semantics_len, idx = batch

        log_probs, pred_len, predictions = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            target_semantics=semantics,
            target_semantics_length=semantics_len,
        )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  # subtract 1 for eos tokens

        loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

        tensorboard_logs = {'train_loss': loss_value.item()}
        if len(self._optimizer.param_groups) == 1:
            tensorboard_logs['learning_rate'] = self._optimizer.param_groups[0]['lr']
        else:
            for i, group in enumerate(self._optimizer.param_groups):
                tensorboard_logs[f'learning_rate_g{i}'] = group['lr']

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=eos_semantics,
                predictions_lengths=pred_len,
                target_lengths=eos_semantics_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict(
        self, input_signal, input_signal_length, processed_signal=None, processed_signal_length=None, dataloader_idx=0
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
        encoded_mask = get_seq_mask(encoded, encoded_len)

        pred_tokens = self.searcher(encoded, encoded_mask)
        predictions = self.decode_semantics(pred_tokens)
        return predictions

    def decode_semantics(self, seq_tokens):
        semantics_str = self.searcher.decode_semantics_from_tokens(seq_tokens)
        return semantics_str

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 4:
            signal, signal_len, semantics, semantics_len = batch
        elif len(batch) == 5:
            signal, signal_len, semantics, semantics_len, sample_id = batch
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, semantics, semantics_len = batch
        else:
            signal, signal_len, transcript, transcript_len, semantics, semantics_len, sample_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, pred_len, predictions = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                target_semantics=semantics,
                target_semantics_length=semantics_len,
            )
        else:
            log_probs, pred_len, predictions = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                target_semantics=semantics,
                target_semantics_length=semantics_len,
            )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  # subtract 1 for bos&eos tokens

        loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

        self._wer.update(
            predictions=predictions,
            targets=eos_semantics,
            predictions_lengths=pred_len,
            target_lengths=eos_semantics_len,
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()

        if self.teacher_force_greedy:
            pred_semantics = []
            true_semantics = []
        else:
            pred_semantics = self.decode_semantics(predictions)
            true_semantics = self.decode_semantics(eos_semantics)

        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
            'pred_semantics': pred_semantics,
            'true_semantics': true_semantics,
        }

    def validation_step_end(self, batch_parts: Dict) -> Optional[Dict]:
        if isinstance(batch_parts['val_loss'], list):
            is_ddp = True
        else:
            is_ddp = False

        results = {
            'val_loss': batch_parts['val_loss'][0] if is_ddp else batch_parts['val_loss'],
            'val_wer_num': batch_parts['val_wer_num'][0] if is_ddp else batch_parts['val_wer_num'],
            'val_wer_denom': batch_parts['val_wer_denom'][0] if is_ddp else batch_parts['val_wer_denom'],
            'val_wer': batch_parts['val_wer'][0] if is_ddp else batch_parts['val_wer'],
            'pred_semantics': [],
            'true_semantics': [],
        }

        if self.teacher_force_greedy:
            return results

        if is_ddp:
            for p in batch_parts['pred_semantics']:
                results['pred_semantics'] += p
            for p in batch_parts['true_semantics']:
                results['true_semantics'] += p
        else:
            results['pred_semantics'] = batch_parts['pred_semantics']
            results['true_semantics'] = batch_parts['true_semantics']

        return results

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum().item()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum().item()
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_wer': wer_num / wer_denom}

        # if not self.teacher_force_greedy:
        #     slurp_evaluator = SLUEvaluator()
        #     for x in outputs:
        #         slurp_evaluator.update(x['pred_semantics'], x['true_semantics'])
        #     slurp_scores = slurp_evaluator.compute()
        #     for k, v in slurp_scores.items():
        #         tensorboard_logs[f"val_{k}"] = v

        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, 'test_wer': wer_num / wer_denom}

        # if not self.teacher_force_greedy:
        #     slurp_evaluator = SLUEvaluator()
        #     for x in outputs:
        #         slurp_evaluator.update(x['pred_semantics'], x['true_semantics'])
        #     slurp_scores = slurp_evaluator.compute()
        #     for k, v in slurp_scores.items():
        #         tensorboard_logs[f"test_{k}"] = v

        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}


class TokenizerBuilder(ASRBPEMixin):
    def __init__(self) -> None:
        self.tokenizer = None
        super().__init__()

    def get_tokenizer(self, cfg: DictConfig) -> 'TokenizerSpec':
        self.tokenizer = None
        self._setup_tokenizer(cfg)
        return copy.deepcopy(self.tokenizer)


# TODO
# class SLUEncDecBPEModel(ASRModel, ExportableEncDecModel, ASRModuleMixin, ASRBPEMixin):
#     """Base class for encoder decoder CTC-based models."""

#     def __init__(self, cfg: DictConfig, trainer: Trainer = None):
#         # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
#         # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
#         self.world_size = 1
#         if trainer is not None:
#             self.world_size = trainer.world_size
#         super().__init__(cfg=cfg, trainer=trainer)

#         self.preprocessor = ModuleBuilder.from_config_dict(self.cfg.preprocessor)
#         self.encoder = ModuleBuilder.from_config_dict(self.cfg.encoder)

#         # Init encoder from SSL checkpoint
#         ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(model_name=self.cfg.ssl_pretrained)
#         self.encoder.load_state_dict(ssl_model.state_dict(), strict=False)
#         del ssl_model

#         if 'tokenizer' not in cfg:
#             raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

#         # Setup the tokenizers for text and semantics
#         # tokenizer_builder = TokenizerBuilder()
#         # self.text_tokenizer = tokenizer_builder.get_tokenizer(cfg.tokenizer.text)
#         # self.semantic_tokenizer = tokenizer_builder.get_tokenizer(cfg.tokenizer.text)
#         self._setup_tokenizer(cfg.tokenizer)
#         self.semantic_tokenizer = self.tokenizer
#         self.text_tokenizer = slu_dataset.EnCharTokenizer(self.semantic_tokenizer)

#         # Initialize a dummy vocabulary
#         vocabulary = self.semantic_tokenizer.tokenizer.get_vocab()
#         vocab_size = len(vocabulary)

#         # Create embedding layer
#         self.cfg.embedding["vocab_size"] = vocab_size
#         cfg.embedding["vocab_size"] = vocab_size
#         self.embedding = ModuleBuilder.from_config_dict(self.cfg.embedding)

#         # Create decoder
#         self.decoder = ModuleBuilder.from_config_dict(self.cfg.decoder)

#         # Create token classifier
#         self.cfg.classifier["num_classes"] = vocab_size
#         cfg.classifier["num_classes"] = vocab_size
#         self.classifier = ModuleBuilder.from_config_dict(self.cfg.claffisier)

#         self.loss = SeqNLLLoss(label_smoothing=getattr(self.cfg, "loss.label_smoothing", 0.0))

#         if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
#             self.spec_augmentation = ModuleBuilder.from_config_dict(self.cfg.spec_augment)
#         else:
#             self.spec_augmentation = None

#         # Setup optional Optimization flags
#         self.setup_optimization_flags()

#         # Adapter modules setup (from ASRAdapterModelMixin)
#         self.setup_adapters()

#         # Setup metric objects
#         self._wer = WERBPE(
#             tokenizer=self.semantic_tokenizer,
#             batch_dim_index=0,
#             use_cer=self.cfg.get('use_cer', False),
#             ctc_decode=True,
#             dist_sync_on_step=True,
#             log_prediction=self.cfg.get("log_prediction", False),
#         )


#     def _setup_dataloader_from_config(self, config: Optional[Dict]):
#         if 'augmentor' in config:
#             augmentor = process_augmentations(config['augmentor'])
#         else:
#             augmentor = None

#         shuffle = config['shuffle']

#         if 'manifest_filepath' in config and config['manifest_filepath'] is None:
#             logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
#             return None

#         dataset = slu_dataset.get_slu_bpe_dataset(
#             config=config,
#             text_tokenizer=self.text_tokenizer,
#             semantic_tokenizer=self.semantic_tokenizer,
#             augmentor=augmentor
#         )

#         if hasattr(dataset, 'collate_fn'):
#             collate_fn = dataset.collate_fn
#         else:
#             collate_fn = dataset.datasets[0].collate_fn

#         return torch.utils.data.DataLoader(
#             dataset=dataset,
#             batch_size=config['batch_size'],
#             collate_fn=collate_fn,
#             drop_last=config.get('drop_last', False),
#             shuffle=shuffle,
#             num_workers=config.get('num_workers', 0),
#             pin_memory=config.get('pin_memory', False),
#         )

#     def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
#         """
#         Sets up the training data loader via a Dict-like object.

#         Args:
#             train_data_config: A config that contains the information regarding construction
#                 of an ASR Training dataset.

#         Supported Datasets:
#             -   :class:`~nemo.collections.asr.data.slu_dataset.AudioTextSemanticsBPEDataset`
#         """
#         if 'shuffle' not in train_data_config:
#             train_data_config['shuffle'] = True

#         # preserve config
#         self._update_dataset_config(dataset_name='train', config=train_data_config)

#         self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

#     def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
#         """
#         Sets up the validation data loader via a Dict-like object.

#         Args:
#             val_data_config: A config that contains the information regarding construction
#                 of an ASR Training dataset.

#         Supported Datasets:
#             -   :class:`~nemo.collections.asr.data.slu_dataset.AudioTextSemanticsBPEDataset`
#         """
#         if 'shuffle' not in val_data_config:
#             val_data_config['shuffle'] = False

#         # preserve config
#         self._update_dataset_config(dataset_name='validation', config=val_data_config)

#         self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

#     def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
#         """
#         Sets up the test data loader via a Dict-like object.

#         Args:
#             test_data_config: A config that contains the information regarding construction
#                 of an ASR Training dataset.

#         Supported Datasets:
#             -   :class:`~nemo.collections.asr.data.slu_dataset.AudioTextSemanticsBPEDataset`
#         """
#         if 'shuffle' not in test_data_config:
#             test_data_config['shuffle'] = False

#         # preserve config
#         self._update_dataset_config(dataset_name='test', config=test_data_config)

#         self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

#     @property
#     def input_types(self) -> Optional[Dict[str, NeuralType]]:
#         if hasattr(self.preprocessor, '_sample_rate'):
#             input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
#         else:
#             input_signal_eltype = AudioSignal()
#         return {
#             "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
#             "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
#             "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
#             "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
#             "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
#         }

#     @property
#     def output_types(self) -> Optional[Dict[str, NeuralType]]:
#         return {
#             "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
#             "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
#             "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
#         }

#     @typecheck()
#     def forward(
#         self, input_signal=None, input_signal_length=None,
#         target_semantics=None, target_semantics_length=None,
#         processed_signal=None, processed_signal_length=None
#     ):
#         """
#         Forward pass of the model.

#         Args:
#             input_signal: Tensor that represents a batch of raw audio signals,
#                 of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
#                 `self.sample_rate` number of floating point values.
#             input_signal_length: Vector of length B, that contains the individual lengths of the audio
#                 sequences.
#             target_semantics: Tensor that represents a batch of semantic tokens,
#                 of shape [B, L].
#             target_semantics_length: Vector of length B, that contains the individual lengths of the semantic
#                 sequences.
#             processed_signal: Tensor that represents a batch of processed audio signals,
#                 of shape (B, D, T) that has undergone processing via some DALI preprocessor.
#             processed_signal_length: Vector of length B, that contains the individual lengths of the
#                 processed audio sequences.

#         Returns:
#             A tuple of 3 elements -
#             1) The log probabilities tensor of shape [B, T, D].
#             2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
#             3) The greedy token predictions of the model of shape [B, T] (via argmax)
#         """
#         has_input_signal = input_signal is not None and input_signal_length is not None
#         has_processed_signal = processed_signal is not None and processed_signal_length is not None
#         if (has_input_signal ^ has_processed_signal) == False:
#             raise ValueError(
#                 f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
#                 " with ``processed_signal`` and ``processed_signal_len`` arguments."
#             )

#         if not has_processed_signal:
#             processed_signal, processed_signal_length = self.preprocessor(
#                 input_signal=input_signal, length=input_signal_length,
#             )

#         if self.spec_augmentation is not None and self.training:
#             processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

#         encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

#         bos_semantics = target_semantics[:, :-1]

#         encoded_mask = get_seq_mask(encoded, encoded_len)
#         bos_semantics_mask = get_seq_mask(bos_semantics, target_semantics_length)

#         decoded = self.decoder(encoder_states=encoded, encoder_mask=encoded_mask, decoder_states=bos_semantics, decoder_mask=bos_semantics_mask)
#         log_probs = self.classifier(decoded)

#         greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

#         return log_probs, encoded_len, greedy_predictions

#     # PTL-specific methods
#     def training_step(self, batch, batch_nb):
#         if len(batch) == 6:
#             signal, signal_len, transcript, transcript_len, semantics, semantics_len = batch
#         else:
#             signal, signal_len, transcript, transcript_len, semantics, semantics_len, idx = batch

#         log_probs, encoded_len, predictions = self.forward(
#             input_signal=signal, input_signal_length=signal_len,
#             target_semantics=semantics, target_semantics_length=semantics_len
#             )

#         eos_semantics = semantics[:,1:]
#         eos_semantics_len = semantics_len - 1

#         loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

#         tensorboard_logs = {'train_loss': loss_value.item(), 'learning_rate': self._optimizer.param_groups[0]['lr']}

#         if hasattr(self, '_trainer') and self._trainer is not None:
#             log_every_n_steps = self._trainer.log_every_n_steps
#         else:
#             log_every_n_steps = 1

#         if (batch_nb + 1) % log_every_n_steps == 0:
#             self._wer.update(
#                 predictions=predictions,
#                 targets=eos_semantics,
#                 target_lengths=eos_semantics_len,
#                 predictions_lengths=encoded_len,
#             )
#             wer, _, _ = self._wer.compute()
#             self._wer.reset()
#             tensorboard_logs.update({'training_batch_wer': wer})

#         return {'loss': loss_value, 'log': tensorboard_logs}

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         signal, signal_len, transcript, transcript_len, sample_id = batch

#         log_probs, encoded_len, predictions = self.forward(
#             input_signal=signal, input_signal_length=signal_len,
#             target_semantics=semantics, target_semantics_length=semantics_len
#             )

#         transcribed_texts = self._wer.ctc_decoder_predictions_tensor(
#             predictions=predictions, predictions_len=encoded_len, return_hypotheses=False,
#         )

#         sample_id = sample_id.cpu().detach().numpy()
#         return list(zip(sample_id, transcribed_texts))

#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         signal, signal_len, transcript, transcript_len = batch

#         log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

#         loss_value = self.loss(
#             log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
#         )
#         self._wer.update(
#             predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
#         )
#         wer, wer_num, wer_denom = self._wer.compute()
#         self._wer.reset()
#         return {
#             'val_loss': loss_value,
#             'val_wer_num': wer_num,
#             'val_wer_denom': wer_denom,
#             'val_wer': wer,
#         }

#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
#         test_logs = {
#             'test_loss': logs['val_loss'],
#             'test_wer_num': logs['val_wer_num'],
#             'test_wer_denom': logs['val_wer_denom'],
#             'test_wer': logs['val_wer'],
#         }
#         return test_logs

#     def test_dataloader(self):
#         if self._test_dl is not None:
#             return self._test_dl

#     def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
#         """
#         Setup function for a temporary data loader which wraps the provided audio file.

#         Args:
#             config: A python dictionary which contains the following keys:
#             paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
#                 Recommended length per file is between 5 and 25 seconds.
#             batch_size: (int) batch size to use during inference. \
#                 Bigger will result in better throughput performance but would use more memory.
#             temp_dir: (str) A temporary directory where the audio manifest is temporarily
#                 stored.
#             num_workers: (int) number of workers. Depends of the batch_size and machine. \
#                 0 - only the main process will load batches, 1 - one worker (not main process)

#         Returns:
#             A pytorch DataLoader for the given audio file(s).
#         """
#         if 'manifest_filepath' in config:
#             manifest_filepath = config['manifest_filepath']
#             batch_size = config['batch_size']
#         else:
#             manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
#             batch_size = min(config['batch_size'], len(config['paths2audio_files']))

#         dl_config = {
#             'manifest_filepath': manifest_filepath,
#             'sample_rate': self.preprocessor._sample_rate,
#             'labels': self.decoder.vocabulary,
#             'batch_size': batch_size,
#             'trim_silence': False,
#             'shuffle': False,
#             'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
#             'pin_memory': True,
#         }

#         temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
#         return temporary_datalayer
