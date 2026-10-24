# Audio Classification (`audio-classification`)

## Overview
Audio Classification assigns categorical labels to raw acoustic waveforms or log-mel spectrogram representations. The model analyzes acoustic frequencies, temporal patterns, and harmonic structures to categorize sound events accurately. Common tasks include acoustic scene analysis, music genre categorization, speaker identification, and environmental sound recognition such as glass breaking, alarms, or sirens. Built using audio spectrogram transformers or specialized convolutional networks, audio classification powers home security monitoring systems, industrial acoustic fault detection, music streaming recommendation engines, speech emotion analytics, and biodiversity wildlife tracking in ecological conservation projects.

## Representative Model: `facebook/mms-lid-1024`

### Model profile
MMS-LID-1024 is a 967M-parameter (F32, ~3.87 GB) language-identification model released in June 2023 under CC-BY-NC-4.0, a non-commercial license. It is fine-tuned from `facebook/mms-1b` and maps raw 16 kHz mono audio to 1024-way ISO 639-3 language logits. It has no text input and does not transcribe.

### Architecture tailored to language ID
It is a wav2vec 2.0 model: seven convolutional feature layers (512 channels, strides 5/2×6, kernels 10/3×5/2×2), 48 transformer layers, 1280 hidden, 16 heads, 128 conv positions, layerdrop 0.05, and a 1024-way classifier projection. There is no adapter (`add_adapter: false`). The self-supervised speech representation plus a single utterance-level classifier is enough for language discrimination without a decoder. See [audio encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
Wav2Vec2's convolutional encoder turns a 16 kHz waveform into ~20 ms frames; the transformer is pretrained by masking frames and contrasting them with quantized targets, then LID fine-tuning pools the sequence and applies a 1024-way linear head. No adapter or decoder is used, so the whole model is an utterance encoder. **Design takeaway:** self-supervised frame features + mean pooling + a linear classifier is the minimal recipe for clip-level audio labels.

### Training and task tailoring
It is fine-tuned from the 1B MMS ASR checkpoint for 1024-language LID, using MMS-lab labeled/unlabeled audio plus FLEURS and VoxLingua-107. The card gives no data sizes or schedules; the paper describes cross-entropy fine-tuning for LID. The classifier is the task-specific adaptation.

### How it is used
Use `AutoFeatureExtractor` + `Wav2Vec2ForSequenceClassification`, resample to 16 kHz mono, and take `argmax(logits, -1)` mapped through `config.id2label`. It runs FP32 on CPU/GPU; there is no official quantized release. It accepts variable-length audio but was trained on utterances, so chunk long recordings.

### Fit and caveats
The tag fits audio classification, but the only task is language ID, not audio-event tagging. The model is non-commercial, has no segmentation/timestamp output, and does not handle code-switching well. The 1024 classes are long-tailed. Reported top-1 is 97.0% on FLEURS and 92.8% on VoxLingua-107 (evaluation covers 102/107 languages); the 126-language model scores 97.5/93.9.

## Related Reading
- [Roadmap: audio encoders](roadmap.md#vision-and-audio-encoders) and [contrastive learning](roadmap.md#contrastive-and-metric-learning).
- [Automatic Speech Recognition](automatic-speech-recognition.md) · [Voice Activity Detection](voice-activity-detection.md) · [Video Classification](video-classification.md)
