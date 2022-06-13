# Model Overview

This collection contains large size versions of Conformer-CTC (around 120M parameters) trained on [MCV-9.0](https://commonvoice.mozilla.org/ca/datasets) dataset.

# Model Architecture
Conformer-CTC model is a non-autoregressive variant of Conformer model [1] for Automatic Speech Recognition which uses CTC loss/decoding instead of Transducer. You may find more info on the detail of this model here: [Conformer-CTC Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html).

# Training
The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml).

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py), and can be found inside the .nemo files.

The vocabulary we use is contains 44 characters:
`['s','e','r','v','i','d','p','o','g','a','m','t','u','l','f','c','z','b','q','n','é',"'",'x','ó','è','h','í','ü','j','à','ï','w','k','y','ç','ú','ò','á','ı','·','ñ','—','–','-']`

Full config can be found inside the .nemo files.

## Datasets
All the models in this collection are trained on MCV-9.0 Catalan dataset, which contains around 1203 hours training, 28 hours of developement and 27 hours of testing speech audios.

# Performance
The list of the available models in this collection is shown in the following table. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding.

| Version | Tokenizer             | Vocabulary Size | Dev WER| Test WER| Train Dataset   |
|---------|-----------------------|-----------------|-----|------|-----------------|
| 1.11.0  | SentencePiece Unigram | 128             |4.70 | 4.27 | MCV-9.0         |


You may use language models (LMs) and beam search to improve the accuracy of the models, as reported in the follwoing table.

| Language Model | Test WER | Test WER w/ Oracle LM | Train Dataset    | Settings                                              |
|----------------|----------|-----------------------|------------------|-------------------------------------------------------|
| N-gram LM      |     3.77 |        1.54           |MCV-9.0 Train set |N=6, beam_width=128, ngram_alpha=1.5, ngram_beta=2.0   |

# How to Use this Model
The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

## Automatically load the model from NGC
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_ca_conformer_ctc_large")
```

## Transcribing text with this model
```python
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
 pretrained_name="stt_ca_conformer_ctc_large" \
 audio_dir="[your audio dir]"
```

## Input
This model accepts 16000 KHz Mono-channel Audio (wav files) as input.

## Output
This model provides transcribed speech as a string for a given audio sample.

# Limitations
Since all models are trained on just MCV-9.0 dataset, the performance of this model might degrade for speech which includes technical terms,or vernacular that the model has not been trained on. The model might also perform worse for accented speech.

# Reference
[1] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)


# Licence

License to use this model is covered by the NGC [TERMS OF USE](https://ngc.nvidia.com/legal/terms) unless another License/Terms Of Use/EULA is clearly specified. By downloading the public and release version of the model, you accept the terms and conditions of the NGC [TERMS OF USE](https://ngc.nvidia.com/legal/terms).
