# Object Detection (`object-detection`)

## Overview
Object Detection is a foundational computer vision task responsible for identifying the presence, class labels, and precise spatial locations of objects within an image. The model outputs rectangular bounding box coordinates alongside categorical confidence scores for every detected entity. Modern object detectors utilize single-stage convolutional networks, anchor-based pipelines, or end-to-end transformer decoders with bipartite matching loss to process multi-scale features efficiently. This technology is vital for real-world automation, powering real-time pedestrian tracking for self-driving vehicles, automated warehouse package sorting, retail self-checkout systems, security monitoring, and wildlife surveillance across diverse outdoor settings.

## Representative Model: `PekingU/rtdetr_r101vd_coco_o365`

### Model profile
RT-DETR with a ResNet-101-vd backbone is a 76.8M-parameter real-time detector (F32, about 307 MB), uploaded in June 2024 under Apache-2.0. It takes a 640×640 RGB image and returns up to 300 COCO-80 boxes with scores, without non-maximum suppression.

### Architecture tailored to real-time detection
A ResNet-101-vd feature extractor feeds a hybrid encoder (AIFI plus CCFF, one layer, 384 dim). Uncertainty-minimal query selection chooses 300 initial queries, and a six-layer deformable decoder (8 heads, 4 sample points, box refinement) predicts boxes directly. Set prediction with bipartite matching removes the NMS post-processing that slows convolutional detectors, which is why it reaches 74 FPS on a T4 at batch 1. See [vision encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
RT-DETR replaces NMS with set prediction: 300 learned queries are matched one-to-one to ground-truth boxes by Hungarian matching, and a six-layer deformable decoder refines each box from four sampled points per head. The hybrid encoder's AIFI mixes intra-scale features with attention while CCFF fuses scales; uncertainty-minimal query selection picks the best initial queries. **Design takeaway:** DETR-style set loss removes NMS, but needs strong query initialization and denoising to converge.

### Training and task tailoring
The model is pretrained on Objects365 and then fine-tuned for 24 COCO epochs (versus 72 from scratch). Losses are VFL/focal (alpha 0.75, gamma 2), L1 box (weight 5), and GIoU (weight 2), with Hungarian matching and 100 denoising queries. Objects365 pretraining is a large part of the final accuracy.

### How it is used
Use `RTDetrImageProcessor` and `RTDetrForObjectDetection`, resize to 640×640 with ImageNet normalization, and call `post_process_object_detection(threshold=0.3)`. The checkpoint is FP32 with no official quantization recipe.

### Fit and caveats
The tag fits, but the vocabulary is closed COCO-80, so there are no text prompts; input is fixed at 640, and the output is boxes only, not masks. Reported COCO val AP is 56.2 (AP50 74.6, AP75 61.3); without Objects365 pretraining, the COCO-only R101 reaches 54.3 AP.

## Related Reading
- [Roadmap: vision encoders](roadmap.md#vision-and-audio-encoders) and [transformers](roadmap.md#transformers-and-attention).
- [Zero-Shot Object Detection](zero-shot-object-detection.md) · [Image Segmentation](image-segmentation.md) · [Mask Generation](mask-generation.md)
