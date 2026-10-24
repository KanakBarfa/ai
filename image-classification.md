# Image Classification (`image-classification`)

## Overview
Image Classification is a fundamental computer vision task focused on assigning one or more predefined category labels to an entire input image based on its visual contents. The model evaluates global visual features, textures, and object silhouettes to output a probability distribution over fixed target classes. Unlike detection or segmentation, classification does not predict localized coordinates or pixel boundaries, but rather identifies the overarching subject or environment depicted. Modern architectures rely on vision transformers and advanced convolutional neural networks. Image classification serves as the foundational building block for automated medical diagnoses, content filtering, species identification, quality control, and visual recommendation systems.

## Representative Model: `pixai-labs/pixai-tagger-v1.0`

### Model profile
PixAI Tagger v1.0 is a 486.3M-parameter FP32 anime image tagger announced in October 2026 under Apache-2.0. It predicts 30,877 sigmoid tags across general (15,043), character (8,308), style (4,917), copyright (2,460), meta (145), and rating (4) categories. It is a multi-label tagger, not a single-label ImageNet classifier.

### Architecture tailored to dense tagging
It uses a ViTDet-style SAM3 image encoder: 32 blocks, 1024 hidden, 16 heads, patch 14, window 24, with global attention at blocks 7/15/23/31. Learned-query pooling plus a linear head predicts all tags; a reconstruction decoder regularizes training. The 1008×1008 input and high-resolution encoder preserve small character details and style cues that a standard 224px classifier would lose. See [vision encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
PixAI uses a ViTDet/SAM3 encoder with window 24 and global attention at blocks 7/15/23/31, so most attention is local and only four blocks mix globally. Learned-query pooling compresses the 72×72 patch grid into a fixed set of tag queries, and a linear sigmoid head covers 30,877 tags; a reconstruction decoder keeps fine character/style detail alive during fine-tuning. **Design takeaway:** for huge multi-label vocabularies, use query pooling + sigmoid, and keep a reconstruction auxiliary to stop the encoder from collapsing.

### Training and task tailoring
The backbone is fine-tuned from SAM3 on PixAI's anime corpus (May 2026 cutoff), with a 300-image admission floor per tag. Training combines multi-label prediction with the reconstruction regularizer; dataset size and loss are not stated. The tagger was also used to curate the Tsubaki.3 dataset.

### How it is used
Call the Transformers pipeline with `trust_remote_code=True`; the model returns results grouped by category. Resize/pad without stretching. Default per-category sigmoid thresholds are 0.17/0.27/0.15/0.24/0.17/0.41. BF16 inference with FP32 sigmoid is recommended; throughput is about 48 images/s at batch 16 on an H100.

### Fit and caveats
The tag fits loosely: this is anime-specific multi-label tagging, not general image classification. Reported full-vocabulary micro F1 is 0.723 and macro F1 0.674; general tags reach 0.6660, characters 0.9242, and styles 0.8143. Thresholds were tuned on the test split, so those figures are optimistic, and the model is not a moderation system.

## Related Reading
- [Roadmap: vision encoders](roadmap.md#vision-and-audio-encoders) and [contrastive learning](roadmap.md#contrastive-and-metric-learning).
- [Zero-Shot Image Classification](zero-shot-image-classification.md) · [Image Feature Extraction](image-feature-extraction.md) · [Image Segmentation](image-segmentation.md)
