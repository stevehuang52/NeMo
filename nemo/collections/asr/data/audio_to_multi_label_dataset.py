from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_multi_label import AudioToMultiLabelDataset, TarredAudioToMultiLabelDataset
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations


def get_audio_multi_label_dataset(cfg: DictConfig) -> AudioToMultiLabelDataset:
    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    dataset = AudioToMultiLabelDataset(
        manifest_filepath=cfg.get("manifest_filepath"),
        sample_rate=cfg.get("sample_rate"),
        labels=cfg.get("labels", None),
        int_values=cfg.get("int_values", False),
        augmentor=augmentor,
        min_duration=cfg.get("min_duration", None),
        max_duration=cfg.get("max_duration", None),
        trim_silence=cfg.get("trim_silence", False),
        is_regression_task=cfg.get("is_regression_task", False),
        delimiter=cfg.get("delimiter", None),
    )
    return dataset


def get_tarred_audio_multi_label_dataset(
    cfg: DictConfig, shuffle_n: int, global_rank: int, world_size: int
) -> TarredAudioToMultiLabelDataset:

    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    tarred_audio_filepaths = cfg['tarred_audio_filepaths']
    manifest_filepaths = cfg['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = cfg.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]

        dataset = TarredAudioToMultiLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            sample_rate=cfg["sample_rate"],
            labels=cfg['labels'],
            shuffle_n=shuffle_n,
            int_values=cfg.get("int_values", False),
            augmentor=augmentor,
            min_duration=cfg.get('min_duration', None),
            max_duration=cfg.get('max_duration', None),
            trim_silence=cfg.get('trim_silence', False),
            is_regression_task=cfg.get('is_regression_task', False),
            delimiter=cfg.get("delimiter", None),
            shard_strategy=cfg.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
        )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=cfg)
