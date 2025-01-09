# Speech Self-Supervised Learning

This directory contains example scripts to self-supervised speech models. 

There are two main types of supported self-supervised learning methods:
- [Wav2vec-BERT](https://arxiv.org/abs/2108.06209): `speech_pre_training.py`
- [NEST](https://arxiv.org/abs/2408.13106): `masked_token_pred_pretrain.py`
    - For downstream tasks that use NEST as multi-layer feature extractor, please refer to `./downstream/speech_classification_mfa_train.py`
    - For extracting multi-layer features from NEST, please refer to `<NEMO ROOT>/scripts/ssl/extract_features.py`
    - For using NEST as weight initialization for downstream tasks, please refer to the usage of the [maybe_init_from_pretrained_checkpoint()](https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/modelPT.py#L1251) function. Generally speaking, you will need to add `++init_from_pretrained.ssl.name="nvidia/ssl_en_nest_large_v1.0"` and `++init_from_pretrained.ssl.include=["encoder"] \` to your bash script.


For their corresponding usage, please refer to the example yaml config:
- Wav2vec-BERT: `examples/asr/conf/ssl/fastconformer/fast-conformer.yaml`
- NEST: `examples/asr/conf/ssl/nest/nest_fast-conformer.yaml`


