from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.generate_data import generate_dataset
from src.multi_classification_models import EncDecMultiClassificationModel

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

pl.seed_everything(0)


@hydra_runner(config_path="./configs", config_name="debug")
def main(cfg):

    data_cfg = cfg.data
    manifest_path = str(Path(data_cfg.root_dir) / Path("synth_manifest.json"))
    # generate_dataset(
    #     data_cfg.root_dir,
    #     data_cfg.num_samples,
    #     data_cfg.sample_duration,
    #     data_cfg.total_duration,
    #     data_cfg.sample_rate,
    # )

    OmegaConf.set_struct(cfg, False)
    cfg.model.train_ds.manifest_filepath = manifest_path
    cfg.model.validation_ds.manifest_filepath = manifest_path
    OmegaConf.set_struct(cfg, True)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecMultiClassificationModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    # if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
    #     if asr_model.prepare_test(trainer):
    #         trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
