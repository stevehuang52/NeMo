from copy import deepcopy
from math import ceil
from typing import Dict, List, Optional, Union

import ipdb
import torch
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from .slu_dataset import DataConstantsSLU as DC
from .slu_dataset import DatumSLU, get_slu_dataset
from .slu_loss import SeqNLLLoss
from .slu_utils import SearcherConfig, SequenceGenerator, get_seq_mask
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models import asr_model
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import Serialization as ModuleBuilder
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging, model_utils


class SLU2NLUEncDecCascadeModel(ModelPT, ASRBPEMixin):
    MODE_SLU = "slu"
    MODE_NLU = "nlu"
    MODE_NLU_ORACLE = "nlu_oracle"

    def __init__(self, cfg: DictConfig, trainer=None):
        self.mode = cfg.get("mode", self.MODE_NLU)
        share_tokenizer = cfg.get("share_tokenizer", False)
        self._setup_tokenizer(cfg.tokenizer)
        self.nlu_tokenizer = self.tokenizer

        if "ctc" in cfg.asr_model:
            asr_model = EncDecCTCModelBPE.from_pretrained(cfg.asr_model)
        else:
            asr_model = EncDecRNNTBPEModel.from_pretrained(cfg.asr_model)

        self.asr_config = deepcopy(asr_model.cfg)
        self.asr_vocab_size = asr_model.cfg.decoder["vocab_size"]
        if share_tokenizer:
            self.asr_tokenizer = self.nlu_tokenizer
        else:
            self.asr_tokenizer = deepcopy(asr_model.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        if self.mode == self.MODE_SLU:
            self.asr_model = asr_model
        else:
            self.asr_model = None
            del asr_model

        self.nlu_vocabulary = self.nlu_tokenizer.tokenizer.get_vocab()
        self.nlu_vocab_size = len(self.nlu_vocabulary)

        # Create embedding layer
        self.cfg.asr_embedding["vocab_size"] = self.asr_vocab_size
        self.asr_embedding = ModuleBuilder.from_config_dict(self.cfg.asr_embedding)

        self.cfg.nlu_embedding["vocab_size"] = self.nlu_vocab_size
        self.embedding = ModuleBuilder.from_config_dict(self.cfg.nlu_embedding)

        # Create decoder
        self.decoder = ModuleBuilder.from_config_dict(self.cfg.decoder)

        # Create token classifier
        self.cfg.classifier["num_classes"] = self.nlu_vocab_size
        self.classifier = ModuleBuilder.from_config_dict(self.cfg.classifier)

        self.loss = SeqNLLLoss(label_smoothing=self.cfg.get("loss.label_smoothing", 0.0))

        self.teacher_force_greedy = self.cfg.searcher.get("teacher_force_greedy", False)
        self.searcher = SequenceGenerator(cfg.searcher, self.embedding, self.decoder, self.classifier, self.tokenizer)

        # Setup metric objects
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def set_decoding_strategy(self, cfg: SearcherConfig):
        max_len = getattr(self.searcher, "generator.max_seq_length", cfg.max_sequence_length)
        max_delta = getattr(self.searcher, "generator.max_delta_length", cfg.max_delta_length)
        cfg.max_sequence_length = max_len
        cfg.max_delta_length = max_delta
        self.searcher = SequenceGenerator(cfg, self.embedding, self.decoder, self.classifier, self.tokenizer)

    def get_asr_predictions(self, signal, signal_len):
        if isinstance(self.asr_model, EncDecCTCModelBPE):
            _, encoded_len, predictions = self.asr_model.forward(input_signal=signal, input_signal_length=signal_len)
            return predictions, encoded_len
        else:
            raise NotImplementedError(f"RNNT model currently not supported")

    def get_ctc_decoded_tokens(self, signal, signal_len):
        _, encoded_len, predictions = self.asr_model.forward(input_signal=signal, input_signal_length=signal_len)
        predicted_text = self.asr_model._wer.ctc_decoder_predictions_tensor(predictions, encoded_len)
        # TODO: tokenize predicted text

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        pred_text=None,
        pred_text_length=None,
        target_semantics=None,
        target_semantics_length=None,
    ):
        if self.asr_model is not None:
            asr_tokens, asr_tokens_len = self.get_asr_predictions(input_signal, input_signal_length)
        else:
            asr_tokens, asr_tokens_len = pred_text, pred_text_length

        asr_embeddings = self.asr_embedding(asr_tokens)

        encoded = asr_embeddings
        encoded_len = asr_tokens_len
        encoded_mask = get_seq_mask(encoded, encoded_len)

        if target_semantics is None:
            predictions, predictions_len = self.searcher(encoded, encoded_mask, return_length=True)
            return None, predictions_len, predictions

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

        predictions_len = self.searcher.get_seq_length(predictions)
        return log_probs, predictions_len, predictions

    # PTL-specific methods
    def training_step(self, batch: DatumSLU, batch_nb: int):
        signal = batch.get(DC.FIELD_AUDIO)
        signal_len = batch.get(DC.FIELD_AUDIO_LEN)
        pred_text = batch.get(DC.FIELD_PRED_TEXT)
        pred_text_len = batch.get(DC.FIELD_PRED_TEXT_LEN)
        semantics = batch.get(DC.FIELD_SEMANTICS)
        semantics_len = batch.get(DC.FIELD_SEMANTICS_LEN)

        ipdb.set_trace()
        if self.mode == self.MODE_NLU_ORACLE:
            pred_text = batch.get(DC.FIELD_TEXT)
            pred_text_len = batch.get(DC.FIELD_TEXT_LEN)

        log_probs, predictions_len, predictions = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            pred_text=pred_text,
            pred_text_length=pred_text_len,
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
                predictions_lengths=predictions_len,
                target_lengths=eos_semantics_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict(
        self,
        input_signal=None,
        input_signal_length=None,
        input_text=None,
        input_text_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        has_audio = has_input_signal or has_processed_signal
        if has_audio and not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if has_audio:
            if self.spec_augmentation is not None and self.training:
                processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
            encoded, encoded_len = self.get_asr_predictions(
                audio_signal=processed_signal, length=processed_signal_length
            )
            encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
        else:
            encoded, encoded_len = input_text, input_text_length

        encoded_mask = get_seq_mask(encoded, encoded_len)

        pred_tokens = self.searcher(encoded, encoded_mask)
        predictions = self.decode_semantics(pred_tokens)
        return predictions

    def decode_semantics(self, seq_tokens):
        semantics_str = self.searcher.decode_semantics_from_tokens(seq_tokens)
        return semantics_str

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal = batch.get(DC.FIELD_AUDIO)
        signal_len = batch.get(DC.FIELD_AUDIO_LEN)
        pred_text = batch.get(DC.FIELD_PRED_TEXT)
        pred_text_len = batch.get(DC.FIELD_PRED_TEXT_LEN)
        semantics = batch.get(DC.FIELD_SEMANTICS)
        semantics_len = batch.get(DC.FIELD_SEMANTICS_LEN)

        if self.mode == self.MODE_NLU_ORACLE:
            pred_text = batch.get(DC.FIELD_TEXT)
            pred_text_len = batch.get(DC.FIELD_TEXT_LEN)

        log_probs, predictions_len, predictions = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            pred_text=pred_text,
            pred_text_length=pred_text_len,
            target_semantics=semantics,
            target_semantics_length=semantics_len,
        )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  # subtract 1 for bos&eos tokens

        loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

        self._wer.update(
            predictions=predictions,
            targets=eos_semantics,
            predictions_lengths=predictions_len,
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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_loss': logs['val_loss'],
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            'test_wer': logs['val_wer'],
        }
        return test_logs

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

        # Instantiate tarred dataset loader or normal dataset loader
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = get_slu_dataset(
            config=config,
            text_tokenizer=self.asr_tokenizer,
            semantic_tokenizer=self.nlu_tokenizer,
            augmentor=augmentor,
        )

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.decoder.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls):
        pass
