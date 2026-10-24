# Image Segmentation (`image-segmentation`)

## Overview
Image Segmentation partitions a digital image into multiple meaningful visual segments by assigning a category label to every individual pixel. This task encompasses semantic segmentation which labels broad categories like road or sky, instance segmentation which distinguishes separate individual objects of the same class, and panoptic segmentation which unifies both approaches. By providing pixel-precise boundaries, segmentation delivers a far richer spatial understanding than standard bounding boxes. This precision makes image segmentation indispensable in mission-critical applications including autonomous vehicle obstacle avoidance, surgical robotic guidance, satellite land-cover mapping, industrial defect detection, and computational photography background replacement.

## Representative Model: `facebook/sapiens-seg-1b-torchscript`

### Model profile
Sapiens-1B segmentation is a 1.169B-parameter human-centric ViT exported as TorchScript (`.pt2`), uploaded in September 2024 under CC-BY-NC-4.0. It maps a 1024×768 RGB image to a per-pixel 28-class human body-part map. It is not a generic semantic/instance segmentation model.

### Architecture tailored to body-part segmentation
The backbone is a ViT-1B with patch 16, 1536 embedding dimension, 40 layers, 24 heads, and a 6144 FFN, plus a lightweight task decoder. Native 1K inference preserves fine boundaries around hair, fingers, and clothing. The model was pretrained with Sapiens' masked autoencoding recipe on 300M in-the-wild human images at 1024×1024 (about 1.2T tokens), then the segmentation head was fine-tuned end-to-end with weighted cross-entropy on 100K images, a 28-class vocabulary, 4:3 resizing, random COCO backgrounds, and layer-wise LR decay 0.85.

### Niche mechanism
Sapiens uses a plain ViT-1B with patch 16 and a lightweight decoder, but the pretraining is domain-specific: MAE on 300M human images at 1K, then end-to-end fine-tuning with weighted CE, random COCO backgrounds, and layer-wise LR decay 0.85. The 4:3 resize and native 1K resolution preserve fingers, hair, and clothing boundaries. **Design takeaway:** a domain-matched MAE backbone + simple decoder beats generic segmentation; resolution is part of the architecture.

### How it is used
This is not a Transformers pipeline. Use the Sapiens repository and the TorchScript `.pt2` with a 1024×768 RGB input. No BF16 or quantized recipe is published, so expect FP32-class memory and compute (4.647 TFLOPs).

### Fit and caveats
The tag implies general image segmentation, but this is human-only semantic parsing; it does not produce instance masks or open-world prompts, the resolution is fixed, and the license is non-commercial. Reported quality is 79.9 mIoU / 89.1 mAcc on the 28-class Humans-2K test, versus 64.1 mIoU for DeepLabV3+; the 2B variant reaches 81.2 mIoU.

## Related Reading
- [Roadmap: vision encoders](roadmap.md#vision-and-audio-encoders) and [Sapiens family](roadmap.md#model-family-notes).
- [Mask Generation](mask-generation.md) · [Keypoint Detection](keypoint-detection.md) · [Depth Estimation](depth-estimation.md)
