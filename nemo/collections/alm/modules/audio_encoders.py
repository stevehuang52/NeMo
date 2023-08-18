from omegaconf import DictConfig

from nemo.core.classes import Exportable, NeuralModule, typecheck


class AudioEncoder(NeuralModule, Exportable):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.from_config_dict(cfg.encoder)
        if hasattr(cfg, 'preprocessor') and cfg.preprocessor is not None:
            self.preprocessor = self.from_config_dict(cfg.preprocessor)
        else:
            self.preprocessor = None

        if hasattr(cfg, 'spec_augment') and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None

    def forward(
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
            if self.preprocessor is None:
                raise ValueError(f"preprocessor cannot be None when has_processed_signal is False")
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0].transpose(1, 2)  # [B, D, T] -> [B, T, D]
        encoded_len = encoder_output[1]
        # audio_feat, audio_feat_len = self.connector(encoded, encoded_len)
        return encoded, encoded_len
