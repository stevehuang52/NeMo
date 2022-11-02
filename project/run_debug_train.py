import enum
from pathlib import Path
from tqdm import tqdm
import os

import pytorch_lightning as pl

pl.seed_everything(0)

import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

from omegaconf import OmegaConf
from src.generate_data import generate_dataset
from src.multi_classification_models import EncDecMultiClassificationModel
from src.audio_to_multi_label import get_audio_multi_label_dataset

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.core import typecheck
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler


@hydra_runner(config_path="./configs", config_name="debug")
def main(cfg):

    data_cfg = cfg.data
    train_data_dir = cfg.data.train_dir
    val_data_dir = cfg.data.val_dir
    if not data_cfg.skip:
        generate_dataset(
            train_data_dir,
            data_cfg.num_samples,
            data_cfg.sample_duration,
            data_cfg.total_duration,
            data_cfg.sample_rate,
        )

        generate_dataset(
            val_data_dir, 100, data_cfg.sample_duration, data_cfg.total_duration, data_cfg.sample_rate,
        )

    if data_cfg.data_only:
        exit(0)

    OmegaConf.set_struct(cfg, False)
    cfg.model.train_ds.manifest_filepath = str(Path(train_data_dir) / Path("synth_manifest.json"))
    cfg.model.validation_ds.manifest_filepath = str(Path(val_data_dir) / Path("synth_manifest.json"))

    if "augmentor" in cfg.model.train_ds and "noise" in cfg.model.train_ds.augmentor:
        cfg.model.train_ds.augmentor.noise.manifest_path = str(Path(val_data_dir) / Path("synth_manifest.json"))
    OmegaConf.set_struct(cfg, True)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    mode = cfg.get("mode", "ptl")
    if mode == "ptl":
        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))
        model = EncDecMultiClassificationModel(cfg=cfg.model, trainer=trainer)

        trainer.fit(model)

        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            if model.prepare_test(trainer):
                trainer.test(model)
    elif mode == "torch_ddp":
        train_with_torch_ddp(cfg)
    else:
        typecheck.disable_checks()
        model = EncDecMultiClassificationModel(cfg=cfg.model)
        model.to("cuda")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
        OmegaConf.set_struct(cfg, False)

        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        train_dl = model._train_dl
        val_dl = model._validation_dl

        optim, _ = model.setup_optimization(cfg.model.optim)

        max_epochs = cfg.trainer.max_epochs
        for i in range(max_epochs):
            logging.info(f"Training epoch {i+1}/{max_epochs}...")
            model.train()
            for batch in tqdm(train_dl, ncols=70, total=len(train_dl), leave=True):
                audio_signal, audio_signal_len, labels, labels_len = batch

                audio_signal = audio_signal.to("cuda")
                audio_signal_len = audio_signal_len.to("cuda")
                labels = labels.to("cuda")
                labels_len = labels_len.to("cuda")

                optim.zero_grad()

                logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                labels, labels_len = model.reshape_labels(logits, labels, labels_len)
                masks = model.get_label_masks(labels, labels_len)

                loss = model.loss(logits=logits, labels=labels, loss_mask=masks)
                loss.backward()

                optim.step()

            logging.info(f"Evaluating epoch {i+1}/{max_epochs}...")
            model.eval()
            for batch in tqdm(val_dl, ncols=70, total=len(train_dl), leave=True):
                audio_signal, audio_signal_len, labels, labels_len = batch

                audio_signal = audio_signal.to("cuda")
                audio_signal_len = audio_signal_len.to("cuda")
                labels = labels.to("cuda")
                labels_len = labels_len.to("cuda")

                logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                labels, labels_len = model.reshape_labels(logits, labels, labels_len)
                masks = model.get_label_masks(labels, labels_len)

                loss = model.loss(logits=logits, labels=labels, loss_mask=masks)


def train_with_torch_ddp(cfg):
    typecheck.disable_checks()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(device_id))

    model = EncDecMultiClassificationModel(cfg=cfg.model)
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    train_dataset = model._train_dl.dataset
    val_dataset = model._validation_dl.dataset
    reshape_labels_fn = model.reshape_labels
    get_label_masks_fn = model.get_label_masks
    loss_fn = model.loss

    optim, _ = model.setup_optimization(cfg.model.optim)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.model.train_ds.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.model.train_ds.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.model.validation_ds.batch_size,
        shuffle=False,
        num_workers=cfg.model.validation_ds.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    max_epochs = cfg.trainer.max_epochs
    for epoch in range(max_epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
        # train for one epoch
        if dist.get_rank() == 0:
            logging.info(f"Training epoch {epoch+1} of {max_epochs}")
        model.train()
        for batch in train_loader:
            audio_signal, audio_signal_len, labels, labels_len = batch
            audio_signal = audio_signal.to(device, non_blocking=True)
            audio_signal_len = audio_signal_len.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_len = labels_len.to(device, non_blocking=True)

            optim.zero_grad()

            logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)
            labels, labels_len = reshape_labels_fn(logits, labels, labels_len)
            masks = get_label_masks_fn(labels, labels_len)

            loss = loss_fn(logits=logits, labels=labels, loss_mask=masks)
            loss.backward()
            optim.step()

        if dist.get_rank() == 0:
            logging.info(f"Evaluating epoch {epoch+1} of {max_epochs}")
        model.eval()
        for batch in val_loader:
            audio_signal, audio_signal_len, labels, labels_len = batch
            audio_signal = audio_signal.to(device, non_blocking=True)
            audio_signal_len = audio_signal_len.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_len = labels_len.to(device, non_blocking=True)

            logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)
            labels, labels_len = reshape_labels_fn(logits, labels, labels_len)
            masks = get_label_masks_fn(labels, labels_len)

            loss = loss_fn(logits=logits, labels=labels, loss_mask=masks)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
