from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import numba
import torch

from src.multi_classification_models import EncDecMultiClassificationModel
from src.audio_to_multi_label import get_audio_multi_label_dataset

train_ds_cfg = {
    "manifest_filepath": "./synth_audio_train/synth_manifest.json",
    "sample_rate": 16000,
    "labels": ['0', '1'],
    "batch_size": 128,
    "shuffle": False,
    "shuffle_n": 2048,
    "num_workers": 8,
    "pin_memory": True,
}
train_ds_cfg = OmegaConf.create(train_ds_cfg)

dataset = get_audio_multi_label_dataset(train_ds_cfg)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=train_ds_cfg.get("batch_size", 1),
    collate_fn=dataset.collate_fn,
    drop_last=train_ds_cfg.get('drop_last', False),
    shuffle=train_ds_cfg.get('shuffle', False),
    num_workers=train_ds_cfg.get('num_workers', 0),
    pin_memory=train_ds_cfg.get('pin_memory', False),
)


for epoch in range(10):
    print(f"Epoch {epoch}/10")
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader), ncols=70)):
        numba.core.entrypoints.init_all()
